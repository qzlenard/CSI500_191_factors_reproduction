"""
[CN] IO 工具契约：目录、原子写、滚动追加、稳健读取。
[Purpose] File I/O helpers with atomic semantics and rolling retention.

Interfaces:
- ensure_dir(path: str) -> None
- write_csv_atomic(path: str, df: pd.DataFrame, index: bool=True) -> None
- append_with_rolloff(path: str, df: pd.DataFrame, key: str, keep_last: int) -> None
  * Appends rows and keeps only the last `keep_last` unique `key` values.
- read_csv_safe(path: str, parse_dates: list[str]|None=None) -> pd.DataFrame

Contracts:
- Atomic write via temp file + rename; create parent dirs if missing.
- Rolloff judged by sorted unique `key` (e.g., date).
"""
from __future__ import annotations

# file: src/utils/fileio.py
# -*- coding: utf-8 -*-
"""
File I/O helpers with atomic semantics and rolling retention.

Public API
----------
- ensure_dir(path, is_file=False) -> Path
- write_csv_atomic(path, df, index=False, encoding="utf-8", float_format=None, **kwargs) -> None
- read_csv_safe(path, parse_dates=None, dtype=None, encoding="utf-8", na_values=None, default=None) -> pd.DataFrame
- append_with_rolloff(path, df_new, key="date", keep_last=252, encoding="utf-8", sort_key_ascending=True) -> pd.DataFrame
- write_text_atomic(path, text, encoding="utf-8") -> None
- read_text_safe(path, encoding="utf-8", default="") -> str

Design notes
------------
* Atomic write: write to a temp file in the same directory, flush+fsync (if available), then os.replace().
* Locking: prefer utils.state.with_file_lock(); fallback is a no-op CM (process-local only).
* Rolloff: keep the last `keep_last` unique values of `key` after concat+dedup (new rows override old on key collision).
* Logging: via utils.logging; falls back to print if logger isn't ready.
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union, Iterable, Dict, Any

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Logging (soft dependency)
# ---------------------------------------------------------------------
try:
    from .logging import step, warn, error, debug  # type: ignore
except Exception:  # pragma: no cover
    def step(msg: str) -> None: print(f"[STEP] {msg}")     # type: ignore
    def warn(msg: str) -> None: print(f"[WARN] {msg}")     # type: ignore
    def error(msg: str) -> None: print(f"[ERROR] {msg}")   # type: ignore
    def debug(msg: str) -> None: print(f"[DEBUG] {msg}")   # type: ignore

# ---------------------------------------------------------------------
# Locking (soft dependency)
# ---------------------------------------------------------------------
try:
    from .state import with_file_lock  # type: ignore
except Exception:
    @contextmanager
    def with_file_lock(_path: Union[str, Path]):  # type: ignore
        """No-op fallback (NOT cross-process safe)."""
        yield

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def ensure_dir(path: Union[str, Path], is_file: bool = False) -> Path:
    """
    Ensure a directory exists. If `is_file=True`, treat `path` as a file path and ensure its parent.
    """
    p = Path(path)
    dir_path = p.parent if is_file else p
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        debug(f"[IO] Created directory: {dir_path}")
    return dir_path


def write_csv_atomic(
    path: Union[str, Path],
    df: pd.DataFrame,
    index: bool = False,
    encoding: str = "utf-8",
    float_format: Optional[str] = None,
    **to_csv_kwargs: Any,
) -> None:
    """
    Atomically write a DataFrame to CSV.

    Implementation details (Windows-safe):
    - Create a temp file alongside the target (mkstemp) and close its raw FD.
    - Re-open the temp file ourselves in **write mode** and pass the handle to `df.to_csv(...)`.
      This guarantees we own the writable descriptor we will `flush + fsync`.
    - `os.replace(tmp, path)` is atomic on POSIX & Windows.
    """
    path = Path(path)
    ensure_dir(path, is_file=True)

    # Create temp file in the same directory for atomic replace
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=str(path.parent))
    # Close the low-level FD returned by mkstemp; we'll reopen in text mode for pandas
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    try:
        # Open in text write mode; newline="" to avoid extra CRLF conversions by csv writer
        with open(tmp_path, "w", encoding=encoding, newline="") as f:
            df.to_csv(
                f,
                index=index,
                float_format=float_format,
                **to_csv_kwargs,
            )
            # Ensure Python buffers are flushed
            f.flush()
            # Best-effort fsync: required for durability on sudden power loss; not all FS support it equally on Win.
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError) as e:  # pragma: no cover
                # On some filesystems/handles (or sandboxed environments) fsync may be unsupported.
                # We warn but continue; os.replace below remains atomic at the filesystem level.
                warn(f"[IO] write_csv_atomic: fsync not supported ({e}); continuing without fsync.")

        # After the handle is closed, perform atomic replacement
        os.replace(tmp_path, path)
        debug(f"[IO] Atomic write OK → {path} (rows={len(df)})")

    except Exception as e:
        # Clean up temp file on failure
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise e


def read_csv_safe(
    path: Union[str, Path],
    parse_dates: Optional[Union[bool, list, dict]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    encoding: str = "utf-8",
    na_values: Optional[Iterable] = None,
    default: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Safely read a CSV. Returns `default` (or empty DataFrame) if file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        debug(f"[IO] read_csv_safe: file not found, returning default → {path}")
        return default.copy() if isinstance(default, pd.DataFrame) else pd.DataFrame()

    with with_file_lock(path):
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                parse_dates=parse_dates,
                dtype=dtype,
                na_values=na_values,
                engine="python",  # tolerant to odd delimiters
            )
            debug(f"[IO] read_csv_safe OK ← {path} (rows={len(df)})")
            return df
        except pd.errors.EmptyDataError:
            warn(f"[IO] read_csv_safe: empty CSV → {path}")
            return pd.DataFrame()
        except Exception as e:
            error(f"[IO] read_csv_safe failed ← {path}: {e}")
            raise


def append_with_rolloff(
    path: Union[str, Path],
    df_new: pd.DataFrame,
    key: str = "date",
    keep_last: int = 252,
    encoding: str = "utf-8",
    sort_key_ascending: bool = True,
) -> pd.DataFrame:
    """
    Append new rows to a CSV (creating it if necessary) and keep only the last `keep_last`
    unique `key` values. On duplicate keys, NEW rows override older ones.
    """
    path = Path(path)
    if key not in df_new.columns:
        raise ValueError(f"`df_new` must contain the key column '{key}'.")

    df_new = df_new.copy()
    if np.issubdtype(df_new[key].dtype, np.datetime64):
        df_new[key] = pd.to_datetime(df_new[key]).dt.strftime("%Y-%m-%d")
    else:
        df_new[key] = df_new[key].astype(str)

    df_old = read_csv_safe(path, parse_dates=None, dtype=None, encoding=encoding)
    rows_old = len(df_old)
    rows_new = len(df_new)

    # Union of columns; prioritize new column order
    all_cols = list(dict.fromkeys(list(df_new.columns) + (list(df_old.columns) if not df_old.empty else [])))
    if not df_old.empty:
        df_old = df_old.reindex(columns=all_cols)
    df_new = df_new.reindex(columns=all_cols)

    df_cat = pd.concat([df_old, df_new], axis=0, ignore_index=True)
    df_cat[key] = df_cat[key].astype(str)

    # Stable sort so 'keep="last"' is deterministic
    df_cat.sort_values(by=[key], ascending=sort_key_ascending, kind="mergesort", inplace=True)

    before_dedup = len(df_cat)
    df_cat = df_cat.drop_duplicates(subset=[key], keep="last")
    deduped = before_dedup - len(df_cat)

    unique_keys = df_cat[key].dropna().unique().tolist()
    if keep_last is not None and keep_last > 0 and len(unique_keys) > keep_last:
        keys_to_keep = unique_keys[-keep_last:] if sort_key_ascending else unique_keys[:keep_last]
        df_out = df_cat[df_cat[key].isin(set(keys_to_keep))].copy()
    else:
        df_out = df_cat

    df_out.sort_values(by=[key], ascending=sort_key_ascending, kind="mergesort", inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    with with_file_lock(path):
        write_csv_atomic(path, df_out, index=False, encoding=encoding)

    kept_rows = len(df_out)
    removed_rows = (rows_old + rows_new) - kept_rows
    step(
        f"[IO] append_with_rolloff → {path.name} | old={rows_old}, new={rows_new}, "
        f"dedup={deduped}, kept={kept_rows}, removed={removed_rows}, keep_last={keep_last}"
    )
    return df_out


def write_text_atomic(path: Union[str, Path], text: str, encoding: str = "utf-8") -> None:
    """
    Atomically write a small text file.
    """
    path = Path(path)
    ensure_dir(path, is_file=True)

    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=str(path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    try:
        with open(tmp_path, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError) as e:  # pragma: no cover
                warn(f"[IO] write_text_atomic: fsync not supported ({e}); continuing without fsync.")
        os.replace(tmp_path, path)
        debug(f"[IO] Atomic text write OK → {path}")
    except Exception as e:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise e


def read_text_safe(path: Union[str, Path], encoding: str = "utf-8", default: str = "") -> str:
    """
    Safely read a text file; return `default` if not exists.
    """
    path = Path(path)
    if not path.exists():
        debug(f"[IO] read_text_safe: file not found, return default → {path}")
        return default
    with with_file_lock(path):
        with open(path, "r", encoding=encoding) as f:
            return f.read()


__all__ = [
    "ensure_dir",
    "write_csv_atomic",
    "append_with_rolloff",
    "read_csv_safe",
    "write_text_atomic",
    "read_text_safe",
]
