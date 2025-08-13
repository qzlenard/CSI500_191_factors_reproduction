# # quick_write_check.py
# import os, tempfile, pathlib, shutil
#
# def check_dir(path):
#     p = pathlib.Path(path)
#     p.mkdir(parents=True, exist_ok=True)  # ensure it exists
#     try:
#         tmp = tempfile.NamedTemporaryFile(dir=p, delete=False)
#         tmp.write(b"ok")
#         tmp.close()
#         os.remove(tmp.name)
#         return True, None
#     except Exception as e:
#         return False, repr(e)
#
# for d in ["out", "data"]:
#     ok, err = check_dir(d)
#     print(f"[CHECK] {d}: {'OK' if ok else 'FAIL'}", ('' if ok else f" -> {err}"))

# # Smoke test for src/utils/logging.py
# from src.utils import logging as log
# import time
#
# log.set_verbosity("DEBUG")
# log_path = log.bind_logfile(prefix="smoke")   # out/logs/ 下会新建文件
#
# log.step("[1/3] Hello logging")
# t0 = time.time()
# for k in range(1, 51):
#     time.sleep(0.01)                          # 模拟耗时任务
#     t0 = log.loop_progress("Demo task", k, 50, start_time=t0, every=10)
#
# log.metric("REG", R2=0.18, N=432, note="smoke")
# try:
#     1/0
# except Exception as e:
#     log.warn("Deliberate warning (non-fatal)", e)
#     log.error("Deliberate error (with stack)", e)
#
# log.done("[3/3] Smoke OK")
# print("LOGFILE =>", log_path)



# # file: tests/smoke_fileio.py
# # -*- coding: utf-8 -*-
# """
# Smoke test for src.utils.fileio
# - Verifies: ensure_dir, write_csv_atomic, append_with_rolloff, read_csv_safe,
#             write_text_atomic, read_text_safe.
# - It writes into out/_smoke_fileio/ and asserts basic invariants.
# Run:
#     python tests/smoke_fileio.py
# or:
#     python -m tests.smoke_fileio
# """
# from __future__ import annotations
#
# import sys
# from pathlib import Path
# import pandas as pd
#
# # Make project root importable when running as a script
# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))
#
# from src.utils.fileio import (  # noqa: E402
#     ensure_dir,
#     write_csv_atomic,
#     append_with_rolloff,
#     read_csv_safe,
#     write_text_atomic,
#     read_text_safe,
# )
#
#
# def main() -> None:
#     smoke_dir = ROOT / "out" / "_smoke_fileio"
#     ensure_dir(smoke_dir / "dummy.txt", is_file=True)
#
#     # --- Text I/O ---
#     txt_path = smoke_dir / "hello.txt"
#     write_text_atomic(txt_path, "hello, fileio")
#     s = read_text_safe(txt_path)
#     assert s == "hello, fileio", "Text roundtrip failed"
#
#     # --- CSV atomic write + safe read ---
#     csv_a = smoke_dir / "a.csv"
#     df_a = pd.DataFrame({"date": ["2025-01-01"], "x": [1]})
#     write_csv_atomic(csv_a, df_a, index=False)
#     df_a_read = read_csv_safe(csv_a)
#     assert len(df_a_read) == 1 and df_a_read.loc[0, "x"] == 1, "CSV atomic write/read failed"
#
#     # --- append_with_rolloff: de-dup by key and keep last K ---
#     roll_path = smoke_dir / "roll.csv"
#     # Seed with two days
#     seed = pd.DataFrame({"date": ["2025-01-01", "2025-01-02"], "v": [10, 20]})
#     write_csv_atomic(roll_path, seed, index=False)
#
#     # New has an overlap on 2025-01-02 (value updated to 21) + a new 2025-01-03
#     new_rows = pd.DataFrame({"date": ["2025-01-02", "2025-01-03"], "v": [21, 30]})
#     out = append_with_rolloff(roll_path, new_rows, key="date", keep_last=2)
#
#     # Expect only last 2 unique dates: 01-02 and 01-03; and value for 01-02 == 21 (new overrides old)
#     keys = list(out["date"])
#     assert keys == ["2025-01-02", "2025-01-03"], f"Rolloff keys wrong: {keys}"
#     v_0102 = out.loc[out["date"] == "2025-01-02", "v"].item()
#     assert v_0102 == 21, "De-dup with new rows overriding failed"
#
#     # read_csv_safe on a missing path returns empty DataFrame
#     missing = read_csv_safe(smoke_dir / "not_exists.csv")
#     assert missing.empty, "read_csv_safe should return empty DataFrame for missing file"
#
#     print("✅ fileio smoke test passed:", smoke_dir)
#
#
# if __name__ == "__main__":
#     main()


# file: tests/smoke_state.py
# -*- coding: utf-8 -*-
"""
Smoke test for src/utils/state.py

What it checks:
  1) Manifest create/save/reload and corruption recovery.
  2) File lock waiting behavior (another thread waits until release).
  3) Stale lock auto-removal.

Run from project root:
    python tests/smoke_state.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import threading
from pathlib import Path

# --- Make sure we can import "src.*" when run as a script --------------------
ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.state import (  # noqa: E402
    load_manifest,
    save_manifest,
    with_file_lock,
)

# --- Test sandbox paths (kept isolated under state/_smoke) -------------------
SMOKE_DIR = Path("state/_smoke")
SMOKE_MANIFEST = SMOKE_DIR / "manifest.json"
SMOKE_LOCKS = SMOKE_DIR / "locks"


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def clean_sandbox() -> None:
    """Remove old smoke artifacts."""
    if SMOKE_DIR.exists():
        for p in sorted(SMOKE_DIR.rglob("*"), reverse=True):
            try:
                if p.is_file():
                    p.unlink(missing_ok=True)
                else:
                    p.rmdir()
            except Exception:
                pass
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_DIR / "locks").mkdir(parents=True, exist_ok=True)


# --- Tests -------------------------------------------------------------------
def test_manifest_lifecycle() -> None:
    print("[SMOKE] test_manifest_lifecycle ...", flush=True)
    clean_sandbox()

    # 1) create (not exists -> defaults)
    m = load_manifest(path=str(SMOKE_MANIFEST), create_if_missing=True)
    assert_true("version" in m and "updated_at" in m, "manifest missing keys after create")

    # 2) mutate & save
    m["version"] = "0.1.0-smoke"
    m["last_processed_date"] = "2024-12-31"
    save_manifest(m, path=str(SMOKE_MANIFEST))

    # 3) reload & verify
    m2 = load_manifest(path=str(SMOKE_MANIFEST), create_if_missing=False)
    assert_true(m2["version"] == "0.1.0-smoke", "manifest save/reload failed (version)")
    assert_true(m2["last_processed_date"] == "2024-12-31", "manifest save/reload failed (date)")
    print("  -> OK")


def test_manifest_corruption_recovery() -> None:
    print("[SMOKE] test_manifest_corruption_recovery ...", flush=True)

    # Corrupt the file with junk
    SMOKE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(SMOKE_MANIFEST, "w", encoding="utf-8") as f:
        f.write("{this is not: valid json}")

    # Load -> should quarantine and recreate defaults
    m3 = load_manifest(path=str(SMOKE_MANIFEST), create_if_missing=True)
    assert_true(isinstance(m3, dict), "manifest recovery failed (not a dict)")
    # Check that a .corrupt.* file exists
    corrupt_candidates = list(SMOKE_MANIFEST.parent.glob(SMOKE_MANIFEST.name + ".corrupt.*"))
    assert_true(len(corrupt_candidates) >= 1, "no quarantined corrupt manifest found")
    print("  -> OK")


def _locker_attempt(result_box: dict) -> None:
    """Worker that tries to acquire the same lock, should block until main releases."""
    try:
        t0 = time.time()
        with with_file_lock(
            lock_name="smoke_lock",
            locks_dir=str(SMOKE_LOCKS),
            timeout_sec=5.0,
            stale_sec=None,  # never steal for this test
            poll_interval=0.05,
        ):
            waited = time.time() - t0
            result_box["waited"] = waited
            result_box["acquired"] = True
    except Exception as e:
        result_box["error"] = repr(e)
        result_box["acquired"] = False


def test_lock_waiting_behavior() -> None:
    print("[SMOKE] test_lock_waiting_behavior ...", flush=True)
    # Main thread acquires the lock and holds it briefly
    with with_file_lock(
        lock_name="smoke_lock",
        locks_dir=str(SMOKE_LOCKS),
        timeout_sec=5.0,
        stale_sec=None,
        poll_interval=0.05,
    ):
        box = {}
        t = threading.Thread(target=_locker_attempt, args=(box,), daemon=True)
        t.start()
        time.sleep(0.8)  # hold lock long enough to force the worker to wait
        # Release context -> worker should acquire afterwards
    # Give the worker a moment to finish
    time.sleep(0.2)
    assert_true(box.get("acquired", False), f"second locker did not acquire (err={box.get('error')})")
    assert_true(box.get("waited", 0.0) >= 0.7, f"second locker did not wait long enough: {box.get('waited')}")
    print("  -> OK")


def test_stale_lock_removal() -> None:
    print("[SMOKE] test_stale_lock_removal ...", flush=True)
    SMOKE_LOCKS.mkdir(parents=True, exist_ok=True)
    stale_path = SMOKE_LOCKS / "stale_case.lock"
    # Create a dummy stale lock
    with open(stale_path, "w", encoding="utf-8") as f:
        f.write("stale")
    # Backdate its mtime to be very old
    old = time.time() - 9999
    os.utime(stale_path, (old, old))

    # Now try to acquire via our context -> it should remove and acquire
    with with_file_lock(
        lock_name="stale_case",
        locks_dir=str(SMOKE_LOCKS),
        timeout_sec=3.0,
        stale_sec=1.0,  # consider stale after 1s
        poll_interval=0.05,
    ):
        pass

    assert_true(not stale_path.exists(), "stale lock was not removed")
    print("  -> OK")


def main() -> int:
    print("=== Running state.py smoke tests ===")
    print(f"Project root: {ROOT}")
    print(f"Sandbox dir : {SMOKE_DIR}")

    try:
        test_manifest_lifecycle()
        test_manifest_corruption_recovery()
        test_lock_waiting_behavior()
        test_stale_lock_removal()
    except AssertionError as e:
        print(f"[SMOKE][FAIL] {e}")
        return 1
    except Exception as e:
        print(f"[SMOKE][ERROR] {e}")
        return 2

    print("[SMOKE] All checks passed ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

