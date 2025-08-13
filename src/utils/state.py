"""
[CN] 状态管理契约：manifest 与文件锁，保证增量与并发安全。
[Purpose] State & lock utilities.

Interfaces:
- load_manifest(path: str) -> dict
- save_manifest(path: str, manifest: dict) -> None
- with_file_lock(lock_path: str):
    '''
    Context manager to ensure single-writer semantics. Implement via OS-level lock or lockfile.
    '''

Manifest Keys (example):
- last_processed_date: "YYYY-MM-DD"
- window_trading_days: int
- version: str
"""
from __future__ import annotations

# file: src/utils/state.py
# -*- coding: utf-8 -*-
"""
State persistence and inter-process file locking.

Public API (kept stable by contract):
    - load_manifest(path: str = "state/manifest.json") -> dict
    - save_manifest(manifest: dict, path: str = "state/manifest.json") -> None
    - with_file_lock(lock_name: str, locks_dir: str = "state/locks", timeout_sec: float = 30.0,
                     stale_sec: float = 600.0, poll_interval: float = 0.2)

Design notes:
    * Cross-platform atomic lock via "create lockfile with O_EXCL".
    * Atomic JSON write using temp file + fsync + os.replace.
    * Defensive handling for corrupted JSON: auto-renames to *.corrupt.<ts>.
    * No external deps; only standard lib + utils.fileio.ensure_dir + utils.logging (if present).
"""

import os
import io
import json
import time
import socket
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Optional

# ---- Optional logging glue (non-fatal if not present) -----------------------
try:
    # Our project logging helpers; all no-ops if not desired.
    from .logging import warn as log_warn, error as log_error, debug as log_debug
except Exception:
    def log_warn(msg: str) -> None:  # type: ignore
        print(f"[WARN] {msg}")

    def log_error(msg: str) -> None:  # type: ignore
        print(f"[ERROR] {msg}")

    def log_debug(msg: str) -> None:  # type: ignore
        # Keep quiet by default for debug to avoid noisy console in production
        pass

# ---- File IO helper (dir ensure) --------------------------------------------
try:
    from .fileio import ensure_dir  # type: ignore
except Exception:
    def ensure_dir(p: os.PathLike | str) -> None:  # Fallback (should not happen)
        Path(p).mkdir(parents=True, exist_ok=True)

# ---- Constants & Defaults ----------------------------------------------------
DEFAULT_MANIFEST: Dict[str, Any] = {
    "last_processed_date": None,   # str in "YYYY-MM-DD" or None
    "window_size": 252,            # rolling days to keep
    "version": "0.1.0",            # pipeline version tag
    "updated_at": None,            # ISO timestamp string
}

DEFAULT_MANIFEST_PATH = "state/manifest.json"
DEFAULT_LOCKS_DIR = "state/locks"


# ---- Internal helpers --------------------------------------------------------
def _now_iso() -> str:
    """Return current time in ISO-8601 (local time with seconds)."""
    # Avoid importing datetime to keep minimal; time.strftime is enough.
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class StatePaths:
    """Resolve and ensure directories for state artifacts."""
    def __init__(self, manifest_path: str = DEFAULT_MANIFEST_PATH, locks_dir: str = DEFAULT_LOCKS_DIR):
        self.manifest_path = Path(manifest_path)
        self.locks_dir = Path(locks_dir)

    def ensure_all(self) -> None:
        ensure_dir(self.manifest_path.parent)
        ensure_dir(self.locks_dir)


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """
    Atomically write JSON to `path`:
        - create temp file in the same directory
        - dump JSON, flush+fsync
        - os.replace to final path
    """
    ensure_dir(path.parent)
    # Use NamedTemporaryFile in target dir for atomic replace on Windows/Unix
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with io.open(tmp_fd, mode="w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)  # atomic
    except Exception as e:
        # Best effort cleanup
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass
        raise e


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_defaults(obj: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-merged dict where missing keys in obj are filled from defaults."""
    out = dict(defaults)
    out.update(obj or {})
    return out


def _pretty_lock_payload() -> str:
    """Return a small JSON string that describes the locker (pid/host/ts)."""
    payload = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at": _now_iso(),
    }
    return json.dumps(payload, ensure_ascii=False)


# ---- Public API: Manifest ----------------------------------------------------
def load_manifest(path: str = DEFAULT_MANIFEST_PATH,
                  create_if_missing: bool = True,
                  defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load state manifest from JSON file. If missing and create_if_missing=True,
    write defaults and return them.

    Args:
        path: Path to manifest JSON.
        create_if_missing: If True (default), create with defaults when not found.
        defaults: Optional default dict to seed missing keys.

    Returns:
        dict with at least keys from DEFAULT_MANIFEST.
    """
    defaults = _merge_defaults(defaults or {}, DEFAULT_MANIFEST)
    p = Path(path)
    if not p.exists():
        if create_if_missing:
            manifest = dict(defaults)
            manifest["updated_at"] = _now_iso()
            _write_json_atomic(p, manifest)
            log_debug(f"[STATE] Created new manifest at {p}")
            return manifest
        else:
            log_warn(f"[STATE] Manifest not found: {p}. Returning defaults (not saved).")
            return dict(defaults)

    try:
        data = _read_json(p)
        manifest = _merge_defaults(data, defaults)
        # Auto-heal: if any required key was missing, rewrite
        if any(k not in data for k in defaults.keys()):
            manifest["updated_at"] = _now_iso()
            _write_json_atomic(p, manifest)
        return manifest
    except Exception as e:
        # Corrupted JSON -> quarantine and reset to defaults
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        corrupt_path = p.with_suffix(p.suffix + f".corrupt.{ts}")
        try:
            os.replace(str(p), str(corrupt_path))
            log_warn(f"[STATE] Corrupted manifest moved to {corrupt_path}. Resetting to defaults.")
        except Exception as e2:
            log_error(f"[STATE] Failed to quarantine corrupted manifest: {e2}")
        manifest = dict(defaults)
        manifest["updated_at"] = _now_iso()
        _write_json_atomic(p, manifest)
        return manifest


def save_manifest(manifest: Dict[str, Any],
                  path: str = DEFAULT_MANIFEST_PATH) -> None:
    """
    Save manifest dict to disk atomically. Updates 'updated_at' automatically.

    Args:
        manifest: Dict-like object.
        path: Path to manifest JSON.
    """
    p = Path(path)
    manifest = dict(manifest)
    manifest["updated_at"] = _now_iso()
    _write_json_atomic(p, manifest)
    log_debug(f"[STATE] Saved manifest -> {p}")


# ---- Public API: File Lock ---------------------------------------------------
@contextmanager
def with_file_lock(lock_name: str,
                   locks_dir: str = DEFAULT_LOCKS_DIR,
                   timeout_sec: float = 30.0,
                   stale_sec: float = 600.0,
                   poll_interval: float = 0.2):
    """
    Inter-process file lock via exclusive lockfile creation.
    Usage:
        with with_file_lock("factor_returns"):
            # critical section

    Args:
        lock_name: A short name for this lock (filename stem).
        locks_dir: Directory to store lock files (will be created).
        timeout_sec: Max time to wait for acquiring the lock; if exceeded -> TimeoutError.
        stale_sec: If an existing lockfile is older than this threshold, it will be considered
                   stale and removed (best-effort). Set to None to never steal locks.
        poll_interval: Sleep interval while waiting.

    Raises:
        TimeoutError: when lock cannot be acquired within timeout.
    """
    locks_dir_p = Path(locks_dir)
    ensure_dir(locks_dir_p)

    lock_path = locks_dir_p / f"{lock_name}.lock"
    deadline = time.monotonic() + float(timeout_sec)

    acquired = False
    last_warn_ts = 0.0

    while True:
        try:
            # O_CREAT|O_EXCL ensures exclusive creation; raise FileExistsError if present.
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    # Write a tiny JSON payload to help debugging orphan locks
                    payload = _pretty_lock_payload()
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                # If writing failed, make sure to remove the lockfile to avoid deadlock.
                try:
                    os.remove(lock_path)
                except Exception:
                    pass
                raise
            acquired = True
            break
        except FileExistsError:
            # If it's stale, try to steal.
            if stale_sec is not None and lock_path.exists():
                try:
                    age = time.time() - lock_path.stat().st_mtime
                except Exception:
                    age = 0.0
                if age > float(stale_sec):
                    try:
                        os.remove(lock_path)
                        log_warn(f"[STATE] Stale lock '{lock_path.name}' (age ~{int(age)}s) removed.")
                        continue  # retry acquire
                    except Exception as e_rm:
                        # Could be a race; just fall through to wait.
                        log_warn(f"[STATE] Failed to remove stale lock '{lock_path.name}': {e_rm}")
            # Wait with periodic gentle logging
            now = time.monotonic()
            if now - last_warn_ts > max(2.0, poll_interval * 5):
                last_warn_ts = now
                log_debug(f"[STATE] Waiting for lock: {lock_path.name} ...")
            if now > deadline:
                raise TimeoutError(f"Timed out acquiring lock '{lock_path.name}' after {timeout_sec}s.")
            time.sleep(float(poll_interval))

    try:
        yield
    finally:
        if acquired:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                # Someone might have cleaned it; ignore.
                pass
            except Exception as e:
                log_warn(f"[STATE] Failed to remove lock '{lock_path}': {e}")


# ---- Convenience helpers (optional, non-contract) ---------------------------
def update_last_processed_date(date_str: Optional[str],
                               path: str = DEFAULT_MANIFEST_PATH) -> Dict[str, Any]:
    """
    Convenience: set 'last_processed_date' and save.
    Returns the updated manifest.
    """
    manifest = load_manifest(path=path, create_if_missing=True)
    manifest["last_processed_date"] = date_str
    save_manifest(manifest, path=path)
    return manifest


# ---- Smoke test --------------------------------------------------------------
if __name__ == "__main__":
    # Minimal self-check without external deps.
    sp = StatePaths()
    sp.ensure_all()

    # Lock smoke test
    try:
        with with_file_lock("state_smoke", stale_sec=5.0, timeout_sec=2.0):
            print("[SMOKE] Acquired lock 'state_smoke'.")
            # Manifest smoke test
            m = load_manifest()
            print(f"[SMOKE] Loaded manifest keys: {sorted(m.keys())}")
            m["version"] = "0.1.0"
            save_manifest(m)
            print("[SMOKE] Saved manifest.")
    except TimeoutError as e:
        print(f"[SMOKE][ERROR] {e}")
