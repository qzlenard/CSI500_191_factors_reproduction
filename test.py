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


# # file: tests/smoke_state.py
# # -*- coding: utf-8 -*-
# """
# Smoke test for src/utils/state.py
#
# What it checks:
#   1) Manifest create/save/reload and corruption recovery.
#   2) File lock waiting behavior (another thread waits until release).
#   3) Stale lock auto-removal.
#
# Run from project root:
#     python tests/smoke_state.py
# """
#
# from __future__ import annotations
#
# import os
# import sys
# import time
# import json
# import threading
# from pathlib import Path
#
# # --- Make sure we can import "src.*" when run as a script --------------------
# ROOT = Path(__file__).resolve().parents[1]  # project root
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))
#
# from src.utils.state import (  # noqa: E402
#     load_manifest,
#     save_manifest,
#     with_file_lock,
# )
#
# # --- Test sandbox paths (kept isolated under state/_smoke) -------------------
# SMOKE_DIR = Path("state/_smoke")
# SMOKE_MANIFEST = SMOKE_DIR / "manifest.json"
# SMOKE_LOCKS = SMOKE_DIR / "locks"
#
#
# def assert_true(cond: bool, msg: str) -> None:
#     if not cond:
#         raise AssertionError(msg)
#
#
# def clean_sandbox() -> None:
#     """Remove old smoke artifacts."""
#     if SMOKE_DIR.exists():
#         for p in sorted(SMOKE_DIR.rglob("*"), reverse=True):
#             try:
#                 if p.is_file():
#                     p.unlink(missing_ok=True)
#                 else:
#                     p.rmdir()
#             except Exception:
#                 pass
#     SMOKE_DIR.mkdir(parents=True, exist_ok=True)
#     (SMOKE_DIR / "locks").mkdir(parents=True, exist_ok=True)
#
#
# # --- Tests -------------------------------------------------------------------
# def test_manifest_lifecycle() -> None:
#     print("[SMOKE] test_manifest_lifecycle ...", flush=True)
#     clean_sandbox()
#
#     # 1) create (not exists -> defaults)
#     m = load_manifest(path=str(SMOKE_MANIFEST), create_if_missing=True)
#     assert_true("version" in m and "updated_at" in m, "manifest missing keys after create")
#
#     # 2) mutate & save
#     m["version"] = "0.1.0-smoke"
#     m["last_processed_date"] = "2024-12-31"
#     save_manifest(m, path=str(SMOKE_MANIFEST))
#
#     # 3) reload & verify
#     m2 = load_manifest(path=str(SMOKE_MANIFEST), create_if_missing=False)
#     assert_true(m2["version"] == "0.1.0-smoke", "manifest save/reload failed (version)")
#     assert_true(m2["last_processed_date"] == "2024-12-31", "manifest save/reload failed (date)")
#     print("  -> OK")
#
#
# def test_manifest_corruption_recovery() -> None:
#     print("[SMOKE] test_manifest_corruption_recovery ...", flush=True)
#
#     # Corrupt the file with junk
#     SMOKE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
#     with open(SMOKE_MANIFEST, "w", encoding="utf-8") as f:
#         f.write("{this is not: valid json}")
#
#     # Load -> should quarantine and recreate defaults
#     m3 = load_manifest(path=str(SMOKE_MANIFEST), create_if_missing=True)
#     assert_true(isinstance(m3, dict), "manifest recovery failed (not a dict)")
#     # Check that a .corrupt.* file exists
#     corrupt_candidates = list(SMOKE_MANIFEST.parent.glob(SMOKE_MANIFEST.name + ".corrupt.*"))
#     assert_true(len(corrupt_candidates) >= 1, "no quarantined corrupt manifest found")
#     print("  -> OK")
#
#
# def _locker_attempt(result_box: dict) -> None:
#     """Worker that tries to acquire the same lock, should block until main releases."""
#     try:
#         t0 = time.time()
#         with with_file_lock(
#             lock_name="smoke_lock",
#             locks_dir=str(SMOKE_LOCKS),
#             timeout_sec=5.0,
#             stale_sec=None,  # never steal for this test
#             poll_interval=0.05,
#         ):
#             waited = time.time() - t0
#             result_box["waited"] = waited
#             result_box["acquired"] = True
#     except Exception as e:
#         result_box["error"] = repr(e)
#         result_box["acquired"] = False
#
#
# def test_lock_waiting_behavior() -> None:
#     print("[SMOKE] test_lock_waiting_behavior ...", flush=True)
#     # Main thread acquires the lock and holds it briefly
#     with with_file_lock(
#         lock_name="smoke_lock",
#         locks_dir=str(SMOKE_LOCKS),
#         timeout_sec=5.0,
#         stale_sec=None,
#         poll_interval=0.05,
#     ):
#         box = {}
#         t = threading.Thread(target=_locker_attempt, args=(box,), daemon=True)
#         t.start()
#         time.sleep(0.8)  # hold lock long enough to force the worker to wait
#         # Release context -> worker should acquire afterwards
#     # Give the worker a moment to finish
#     time.sleep(0.2)
#     assert_true(box.get("acquired", False), f"second locker did not acquire (err={box.get('error')})")
#     assert_true(box.get("waited", 0.0) >= 0.7, f"second locker did not wait long enough: {box.get('waited')}")
#     print("  -> OK")
#
#
# def test_stale_lock_removal() -> None:
#     print("[SMOKE] test_stale_lock_removal ...", flush=True)
#     SMOKE_LOCKS.mkdir(parents=True, exist_ok=True)
#     stale_path = SMOKE_LOCKS / "stale_case.lock"
#     # Create a dummy stale lock
#     with open(stale_path, "w", encoding="utf-8") as f:
#         f.write("stale")
#     # Backdate its mtime to be very old
#     old = time.time() - 9999
#     os.utime(stale_path, (old, old))
#
#     # Now try to acquire via our context -> it should remove and acquire
#     with with_file_lock(
#         lock_name="stale_case",
#         locks_dir=str(SMOKE_LOCKS),
#         timeout_sec=3.0,
#         stale_sec=1.0,  # consider stale after 1s
#         poll_interval=0.05,
#     ):
#         pass
#
#     assert_true(not stale_path.exists(), "stale lock was not removed")
#     print("  -> OK")
#
#
# def main() -> int:
#     print("=== Running state.py smoke tests ===")
#     print(f"Project root: {ROOT}")
#     print(f"Sandbox dir : {SMOKE_DIR}")
#
#     try:
#         test_manifest_lifecycle()
#         test_manifest_corruption_recovery()
#         test_lock_waiting_behavior()
#         test_stale_lock_removal()
#     except AssertionError as e:
#         print(f"[SMOKE][FAIL] {e}")
#         return 1
#     except Exception as e:
#         print(f"[SMOKE][ERROR] {e}")
#         return 2
#
#     print("[SMOKE] All checks passed ✅")
#     return 0
#
#
# if __name__ == "__main__":
#     raise SystemExit(main())


# # file: sanity_numerics.py
# # -*- coding: utf-8 -*-
# """
# Lightweight sanity checks for src/utils/numerics.py
#
# Run:
#     python sanity_numerics.py
#
# This script validates:
# 1) zscore() -> per-column mean≈0, std≈1 ; and group-wise version respects groups.
# 2) winsorize(by=...) -> per-group min/max are inside the group's quantile bounds.
# 3) ridge_fit(alpha≈0) ≈ OLS (safe_lstsq on X with intercept).
# 4) weighted_r2() basics; sample_size_ok(); standardize_exposures(); clip_inf_nan().
# """
#
# from __future__ import annotations
# import pandas as pd
# # --- pytest fixtures & path bootstrap (put at top of test.py) ---
# from pathlib import Path
# import sys
# SRC = Path(__file__).resolve().parent / "src"
# if str(SRC) not in sys.path:
#     sys.path.insert(0, str(SRC))  # ensure 'src' is importable
#
# import pytest
# import numpy as np
#
# @pytest.fixture(scope="function")
# def rng():
#     """Deterministic RNG for tests."""
#     return np.random.default_rng(20250814)
#
#
# # Allow "from utils.numerics import ..." without packaging
# ROOT = Path(__file__).resolve().parent
# SRC = ROOT / "src"
# if str(SRC) not in sys.path:
#     sys.path.insert(0, str(SRC))
#
# from utils.numerics import (  # type: ignore
#     zscore,
#     winsorize,
#     standardize_exposures,
#     ridge_fit,
#     safe_lstsq,
#     add_constant,
#     weighted_r2,
#     sample_size_ok,
#     clip_inf_nan,
# )
#
#
# def _ok(flag: bool, msg: str) -> None:
#     if flag:
#         print(f"[PASS] {msg}")
#     else:
#         print(f"[FAIL] {msg}")
#         raise AssertionError(msg)
#
#
# def test_zscore_basic(rng: np.random.Generator) -> None:
#     n, p = 500, 4
#     X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"c{i}" for i in range(p)])
#     # Inject a few NaNs
#     X.iloc[::73, 0] = np.nan
#     Z = zscore(X)
#
#     means = Z.mean(skipna=True).abs()
#     stds = (Z.std(ddof=0, skipna=True) - 1.0).abs()
#     _ok(bool((means < 1e-10).all()), "zscore(): column means ~ 0")
#     _ok(bool((stds < 1e-10).all()), "zscore(): column std ~ 1 (ddof=0)")
#
#     # Group-wise check
#     by = pd.Series(np.where(np.arange(n) % 2 == 0, "A", "B"), index=X.index)
#     Zg = zscore(X, by=by)
#     for g, sub in Zg.groupby(by):
#         m = sub.mean().abs()
#         s = (sub.std(ddof=0) - 1.0).abs()
#         _ok(bool((m < 1e-10).all()), f"zscore(by): group {g} means ~ 0")
#         _ok(bool((s < 1e-10).all()), f"zscore(by): group {g} std ~ 1")
#
#
# def test_winsorize_group(rng: np.random.Generator) -> None:
#     n, p = 400, 3
#     # Two groups with different scales + outliers
#     g = np.where(np.arange(n) % 2 == 0, "A", "B")
#     A = rng.normal(loc=0.0, scale=1.0, size=(n // 2, p))
#     B = rng.normal(loc=0.0, scale=3.0, size=(n - n // 2, p))
#     X = pd.DataFrame(np.vstack([A, B]), columns=[f"f{i}" for i in range(p)])
#     by = pd.Series(g, index=X.index)
#
#     # add extreme outliers
#     X.iloc[5, 0] = 50.0
#     X.iloc[7, 1] = -60.0
#     X.iloc[-3, 2] = 80.0
#
#     lower, upper = 0.05, 0.95
#     W = winsorize(X, lower=lower, upper=upper, by=by)
#
#     # For each group g and column c, min/max must lie within group's original [q05, q95]
#     for grp, sub in X.groupby(by):
#         q = sub.quantile([lower, upper], interpolation="linear")
#         lo, hi = q.loc[lower], q.loc[upper]
#         Wg = W.loc[sub.index]
#         for c in X.columns:
#             wmin = float(Wg[c].min())
#             wmax = float(Wg[c].max())
#             _ok(wmin >= float(lo[c]) - 1e-12, f"winsorize(by): group {grp} col {c} min >= q{int(lower*100)}")
#             _ok(wmax <= float(hi[c]) + 1e-12, f"winsorize(by): group {grp} col {c} max <= q{int(upper*100)}")
#
#
# def test_ridge_vs_ols(rng: np.random.Generator) -> None:
#     n, p = 300, 6
#     X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
#     beta_true = rng.normal(size=p)
#     y = pd.Series(X.values @ beta_true + rng.normal(scale=0.1, size=n), name="y")
#
#     # Ridge with very small alpha (no intercept penalty)
#     coef_r, intercept_r, info = ridge_fit(X, y, alpha=1e-10, fit_intercept=True, penalize_intercept=False)
#
#     # OLS via lstsq on augmented X
#     X_aug = add_constant(X, name="Intercept")
#     coef_ols, _rank = safe_lstsq(X_aug, y)
#     # Compare
#     diff_beta = (coef_r.values - coef_ols[coef_r.index].values)
#     diff_b0 = abs(intercept_r - float(coef_ols["Intercept"]))
#     _ok(float(np.max(np.abs(diff_beta))) < 5e-6, "ridge_fit(alpha→0) ≈ OLS coefficients")
#     _ok(diff_b0 < 5e-6, "ridge_fit(alpha→0) ≈ OLS intercept")
#     _ok(0.95 <= info["r2"] <= 1.0, "ridge_fit(): high R² on synthetic data")
#
#
# def test_metrics_and_utils() -> None:
#     # weighted_r2 perfect fit -> 1.0
#     y = pd.Series([1.0, 2.0, 3.0, 4.0])
#     yhat = pd.Series([1.0, 2.0, 3.0, 4.0])
#     w = pd.Series([1.0, 0.5, 2.0, 1.0])
#     r2 = weighted_r2(y, yhat, w)
#     _ok(abs(r2 - 1.0) < 1e-12, "weighted_r2(): perfect fit equals 1")
#
#     # sample_size_ok
#     _ok(sample_size_ok(120, 100, margin=5) is True, "sample_size_ok(): sufficient")
#     _ok(sample_size_ok(101, 100, margin=5) is False, "sample_size_ok(): insufficient")
#
#     # standardize_exposures: finite values and clip
#     df = pd.DataFrame({"a": [1, 2, 1000, -999, 3], "b": [5, 4, 3, 2, 1]})
#     Z = standardize_exposures(df, winsor=(0.01, 0.99), clip_z=3.0)
#     _ok(np.isfinite(Z.values).all(), "standardize_exposures(): finite values after pipeline")
#     _ok((np.abs(Z.values) <= 3.0000001).all(), "standardize_exposures(): clip_z respected")
#
#     # clip_inf_nan
#     s = pd.Series([1.0, np.inf, -np.inf, np.nan, 5.0])
#     s2 = clip_inf_nan(s, fill_nan=0.0)
#     _ok(s2.tolist() == [1.0, 0.0, 0.0, 0.0, 5.0], "clip_inf_nan(): inf->NaN->filled")
#
#
# def main() -> None:
#     rng = np.random.default_rng(20250814)
#     test_zscore_basic(rng)
#     test_winsorize_group(rng)
#     test_ridge_vs_ols(rng)
#     test_metrics_and_utils()
#     print("\nAll sanity checks passed ✅")
#
#
# if __name__ == "__main__":
#     main()
#
#
# # quick REPL check
# import pandas as pd
# from src.utils.filters import tradable_codes
#
# df = pd.DataFrame({
#     "date": ["2025-08-13"],
#     "code": ["X"],
#     "close": [10.5],      # 未触及涨停
#     "preclose": [10.0],
#     "volume": [1000],
#     "paused": [0]
# })
# print(tradable_codes(df, "2025-08-13", ["X"]))  # 期望: ['X']


# file: smoke/smoke_myquant_io.py
# -*- coding: utf-8 -*-
# """
# Smoke test for src/api/myquant_io.py using gm.api (掘金GM).
#
# What it does
# ------------
# 1) Pull recent trading dates and pick a short window (default: last 15 trade days).
# 2) Get CSI500 members on the last trade date (configurable index).
# 3) Fetch OHLCV for a small sample of symbols within the window and validate schema.
# 4) Fetch fundamentals snapshot (publication-aware with lag) for the same sample.
# 5) Save small CSV samples to out/logs/ and print concise progress.
#
# How to run
# ----------
# $ python smoke/smoke_myquant_io.py --index SHSE.000905 --days 15 --n 10 --fq pre --lag 30
#
# First-time setup
# ----------------
# - pip install gm
# - In config.py, set: GM_TOKEN = "<your_token>"
#   (Alternatively, export environment variable GM_TOKEN)
# """
# from __future__ import annotations
#
# import argparse
# import os
# import sys
# from datetime import datetime, timedelta
# from typing import List
#
# import pandas as pd
#
# # Make project importable when running from repo root
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
# if PROJ_ROOT not in sys.path:
#     sys.path.insert(0, PROJ_ROOT)
#
# # Import the adapter under test
# from src.api.myquant_io import (  # noqa: E402
#     get_trade_days,
#     get_index_members,
#     get_ohlcv,
#     get_fundamentals_snapshot,
# )
#
# # Prefer our logging helper if available
# try:
#     from src.utils.logging import step, loop_progress, done, warn, error, set_verbosity, bind_logfile  # noqa: E402
# except Exception:  # fallback
#     def step(msg: str) -> None: print(f"[STEP] {msg}")
#     def loop_progress(msg: str) -> None: print(f"[LOOP] {msg}")
#     def done(msg: str = "done") -> None: print(f"[OK] {msg}")
#     def warn(msg: str) -> None: print(f"[WARN] {msg}")
#     def error(msg: str) -> None: print(f"[ERROR] {msg}")
#     def set_verbosity(level: str) -> None: pass
#     def bind_logfile(path: str) -> None: pass
#
#
# REQUIRED_OHLCV_COLS = {
#     "date", "code", "open", "high", "low", "close", "volume",
#     "amount", "preclose", "paused", "high_limit", "low_limit"
# }
#
#
# def ensure_dir(path: str) -> None:
#     os.makedirs(path, exist_ok=True)
#
#
# def pick_trade_window(days: int) -> tuple[str, str, List[pd.Timestamp]]:
#     today = datetime.now().date()
#     start_guess = (today - timedelta(days=max(40, int(days) * 3))).strftime("%Y-%m-%d")
#     end_guess = today.strftime("%Y-%m-%d")
#     tds = get_trade_days(start_guess, end_guess)
#     if len(tds) < days:
#         raise RuntimeError(f"Not enough trade days in range [{start_guess},{end_guess}], got {len(tds)} < {days}")
#     window = tds[-days:]
#     return window[0].strftime("%Y-%m-%d"), window[-1].strftime("%Y-%m-%d"), tds
#
#
# def main() -> int:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--index", type=str, default="SHSE.000905", help="Index code for membership (e.g., SHSE.000905/CSI500)")
#     ap.add_argument("--days", type=int, default=15, help="Number of most-recent trade days to test")
#     ap.add_argument("--n", type=int, default=10, help="Number of symbols to sample for OHLCV/fundamentals")
#     ap.add_argument("--fq", type=str, default="pre", choices=["pre", "post", "none"], help="Adjustment for OHLCV")
#     ap.add_argument("--lag", type=int, default=30, help="Lag days for publication-aware fundamentals")
#     ap.add_argument("--log", type=str, default="out/logs/smoke_myquant_io.log", help="Log file path")
#     args = ap.parse_args()
#
#     # Logging setup
#     ensure_dir(os.path.dirname(args.log))
#     try:
#         set_verbosity("STEP")
#         bind_logfile(args.log)
#     except Exception:
#         pass
#
#     step("Smoke | Step 0: Check GM token presence (config/env)")
#     # Non-fatal check: if token missing, GM calls may fail with auth error; we hint how to fix.
#     token = None
#     try:
#         from config import GM_TOKEN  # type: ignore
#         token = GM_TOKEN
#     except Exception:
#         token = os.environ.get("GM_TOKEN")
#     if not token:
#         warn("GM_TOKEN missing. Set config.GM_TOKEN or env GM_TOKEN before running in production.")
#
#     # 1) Trading window
#     step("Smoke | Step 1: Resolve trade window")
#     start, end, all_tds = pick_trade_window(args.days)
#     last_trade_day = pd.Timestamp(all_tds[-1])
#     done(f"Trade window: {start} -> {end} (last={last_trade_day.date()})")
#
#     # 2) Index members
#     step(f"Smoke | Step 2: Get members of {args.index} @ {last_trade_day.date()}")
#     members = get_index_members(args.index, last_trade_day)
#     if not members:
#         raise RuntimeError("Empty index members. Check index code or network/token.")
#     done(f"Members = {len(members)}")
#
#     # Sample N symbols (stable slice)
#     sample_syms = members[: max(1, args.n)]
#     step(f"Sample {len(sample_syms)} symbols: {', '.join(sample_syms[:5])}{' ...' if len(sample_syms) > 5 else ''}")
#
#     # 3) OHLCV
#     step(f"Smoke | Step 3: Fetch OHLCV for {len(sample_syms)} symbols, fq={args.fq}, {start}->{end}")
#     ohlcv = get_ohlcv(sample_syms, start, end, args.fq)
#     if ohlcv.empty:
#         raise RuntimeError("OHLCV returned empty DataFrame.")
#     missing = REQUIRED_OHLCV_COLS - set(ohlcv.columns)
#     if missing:
#         warn(f"OHLCV missing columns: {sorted(missing)} (filters will fallback for limit prices if absent)")
#     # Basic consistency checks
#     assert ohlcv["code"].nunique() <= len(sample_syms)
#     assert ohlcv["date"].dt.normalize().equals(ohlcv["date"])
#     # Save sample
#     ensure_dir("out/logs")
#     ohlcv.head(50).to_csv("out/logs/smoke_ohlcv_head.csv", index=False)
#     done(f"OHLCV rows={len(ohlcv)}, saved head to out/logs/smoke_ohlcv_head.csv")
#
#     # 4) Fundamentals snapshot
#     step(f"Smoke | Step 4: Fundamentals snapshot @ {last_trade_day.date()} (lag={args.lag}d)")
#     funda = get_fundamentals_snapshot(last_trade_day, sample_syms, args.lag)
#     if funda is None or funda.empty:
#         warn("Fundamentals snapshot empty (plan/version may not have fields).")
#     else:
#         # Save sample
#         funda.reset_index().head(50).to_csv("out/logs/smoke_funda_head.csv", index=False)
#         done(f"Fundamentals cols={list(funda.columns)}, names={len(funda)}; saved head to out/logs/smoke_funda_head.csv")
#
#     done("Smoke test PASSED")
#     return 0
#
#
# if __name__ == "__main__":
#     try:
#         sys.exit(main())
#     except AssertionError as e:
#         error(f"Assertion failed: {e}")
#         sys.exit(2)
#     except Exception as e:
#         # Provide actionable hints for first-time users
#         error(f"Smoke test FAILED: {e}")
#         hint = (
#             "Quick checklist:\n"
#             "1) `pip install gm` (ensure gm.api importable)\n"
#             "2) Set config.GM_TOKEN = '<your_token>' or export env GM_TOKEN\n"
#             "3) Network reachable to GM servers\n"
#             "4) Index code valid (e.g., SHSE.000905)\n"
#         )
#         print(hint)
#         sys.exit(3)
#
# tools/lint_191_yaml.py
# coding: utf-8
# import re, sys, csv, os, io
# from pathlib import Path
#
# YAML_PATH = r"D:\Code\R\quant-leiying\task5\191multi-factor reproduce\data\ref\191factors\191.yaml"
# OUT_CSV   = Path(__file__).with_name("lint_report.csv")
#
# def load_yaml_text(p: str) -> dict:
#     import yaml
#     with open(p, "r", encoding="utf-8") as f:
#         raw = yaml.safe_load(f)
#     if isinstance(raw, list):
#         m = {}
#         for it in raw:
#             fid = int(it.get("id") or re.findall(r"\d+", str(it.get("name","")))[0])
#             m[fid] = it.get("expr","")
#         return m
#     elif isinstance(raw, dict):
#         m = {}
#         for k, v in raw.items():
#             fid = int(re.findall(r"\d+", str(k))[0])
#             m[fid] = v["expr"] if isinstance(v, dict) else str(v)
#         return m
#     else:
#         raise ValueError("Unsupported YAML shape")
#
# # ---- heuristic preprocess, returns (fixed, notes) ----
# def preprocess(expr: str):
#     s0 = s = str(expr)
#     notes = []
#
#     # 1) normalize whitespace & ascii
#     s = s.replace("，", ",").replace("（", "(").replace("）", ")").replace("：", ":")
#     s = s.replace("；", ";")
#
#     # 2) aliases / typos
#     rep = {
#         "DECAYLINEAR": "DECAY_LINEAR",
#         "COVIANCE": "COV",
#         "SMEAN(": "SMA(",
#         "MA(": "SMA(",
#         "DELAT(": "DELTA(",
#         "HGIH": "HIGH",
#         "LOWSMA(": "LOW - SMA(",
#         "BANCHMARKINDEX": "BENCHMARKINDEX",  # 兼容两边拼写
#     }
#     for a,b in rep.items():
#         if a in s:
#             s = s.replace(a,b); notes.append(f"alias:{a}->{b}")
#
#     # 3) words -> operators
#     def sub_word(pat, repl, tag):
#         nonlocal s;
#         if re.search(pat, s, flags=re.I):
#             s = re.sub(pat, repl, s, flags=re.I); notes.append(tag)
#     sub_word(r"\bAND\b", "&", "logic:AND->&")
#     sub_word(r"\bOR\b",  "|", "logic:OR->|")
#     sub_word(r"\bNOT\b", "~", "logic:NOT->~")
#     s = s.replace("||","|").replace("&&","&")
#     s = re.sub(r'!(?!=)', "~", s)
#
#     # 4) power
#     if "^" in s:
#         s = s.replace("^", "**"); notes.append("pow:^->**")
#
#     # 5) single '=' to '==' (but keep <=,>=,!=,==)
#     if re.search(r'(?<![!<>=])=(?![=])', s):
#         s = re.sub(r'(?<![!<>=])=(?![=])', "==", s)
#         notes.append("cmp:=->==")
#
#     # 6) ternary  A?B:C  -> IF(A,B,C)   (non-nested heuristic)
#     if "?" in s and ":" in s:
#         # very simple split (non-nested)
#         m = re.match(r"^(.*)\?(.*):(.*)$", s)
#         if m:
#             cond, a, b = [x.strip() for x in m.groups()]
#             s = f"IF({cond}, {a}, {b})"
#             notes.append("ternary->IF()")
#
#     # 7) SMA(x,n,m) accept; nothing to change, but mark
#     if re.search(r"\bSMA\s*\([^,]+,\s*[^,]+,\s*[^)]+\)", s):
#         notes.append("SMA3")
#
#     # 8) suspicious commas: missing comma between args like A B
#     if re.search(r"\b\w+\s*\(\s*[^,()]+?\s+[^,()]+?\)", s):
#         notes.append("maybe-missing-comma")
#
#     # 9) unbalanced parentheses
#     if s.count("(") != s.count(")"):
#         notes.append("unbalanced-paren")
#
#     # 10) unknown tokens quick scan (very conservative)
#     tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)
#     whitelist = set("""
#         OPEN HIGH LOW CLOSE VWAP VOLUME VOL AMOUNT AMT O H L C V PREV_CLOSE PREVCLOSE PRECLOSE
#         BENCHMARKINDEXOPEN BENCHMARKINDEXHIGH BENCHMARKINDEXLOW BENCHMARKINDEXCLOSE BENCHMARKINDEXVOLUME
#         RET DTM DBM TR HD LD
#         DELAY DELTA SUM MEAN STD VAR TSMAX TSMIN SMA WMA DECAY_LINEAR COUNT PROD SUMIF FILTER
#         HIGHDAY LOWDAY CORR COV REGBETA SEQUENCE RANK TSRANK IF ABS SIGN LOG EXP MAX MIN POWER SIGNED_POWER CLIP
#         REGRESI RESI
#     """.split())
#     unk = sorted(set(t for t in tokens if t.upper() not in whitelist and not t.isupper() and not t.islower()))
#     if unk:
#         notes.append("maybe-unknown:" + "|".join(unk[:5]))
#
#     return s, notes
#
# def main():
#     data = load_yaml_text(YAML_PATH)
#     rows = [("id","raw","fixed","notes")]
#     bad = 0
#     for fid, expr in sorted(data.items(), key=lambda x:x[0]):
#         fixed, notes = preprocess(expr)
#         note = ";".join(notes)
#         # very rough python eval sanity (syntax only)
#         try:
#             compile(fixed, "<expr>", "eval")
#         except SyntaxError as e:
#             bad += 1
#             note = f"SYNTAX:{e.msg};" + note
#         rows.append((fid, expr, fixed, note))
#     with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
#         csv.writer(f).writerows(rows)
#     print(f"Done. total={len(data)} bad={bad} -> {OUT_CSV}")
#
# if __name__ == "__main__":
#     main()

# from src.trading_calendar import is_trading_day
# print("2020-12-29", is_trading_day("2020-12-29"))
# print("2021-01-04", is_trading_day("2021-01-04"))
# print("2024-01-02", is_trading_day("2024-01-02"))

# from src.utils.logging import set_verbosity
# set_verbosity("STEP")  # 或 "DEBUG" 看更详细日志
#
# from src.trading_calendar import get_trading_days
# from src.universe import csi500
#
# days = get_trading_days("2024-01-02", "2024-01-10")
# print("Days:", [d.date() for d in days])
#
# for d in days:
#     codes = csi500(d)          # 这里会自动写入缓存 CSV
#     print(d.date(), len(codes))
#
# # 看看文件是否出现：
# import glob
# print(sorted(glob.glob("data/ref/index_members/*.csv"))[:5])
# #

# from src.universe import csi500
# import pandas as pd
# d = pd.Timestamp("2024-01-02")
# codes = csi500(d)
# print(type(codes), len(codes) if isinstance(codes, list) else codes)
# assert isinstance(codes, list), f"csi500 returned {type(codes)} on {d}"

# # file: test.py
# """
# Purpose
# -------
# Quick probe to report which style subfactors are blocked due to missing
# fundamental fields and to quantify coverage for each required base field.
#
# Usage
# -----
# python test.py --date 2025-08-30 --lag 5 --index SHSE.000905 --verbosity STEP
#
# Notes
# -----
# - This script does NOT change any project files. It's a read-only diagnostics.
# - It normalizes security codes to '######.(SH|SZ)' to match the main pipeline.
# """
#
# from __future__ import annotations
#
# import argparse
# import os
# import sys
# from typing import Dict, List, Tuple
#
# import numpy as np
# import pandas as pd
#
# # Make sure project root is in sys.path when running from repo root
# ROOT = os.path.abspath(os.path.dirname(__file__))
# if ROOT not in sys.path:
#     sys.path.append(ROOT)
#
# # Project utils & APIs
# from src.utils.logging import set_verbosity, step, done, warn, error  # type: ignore
# from src.api.myquant_io import (  # type: ignore
#     get_trade_days,
#     get_index_members,
#     get_fundamentals_snapshot,
# )
#
# # Optional: LAG default from config (soft dependency)
# DEFAULT_LAG = 5
# try:
#     from src.config import CFG  # type: ignore
#     DEFAULT_LAG = int(getattr(getattr(CFG, "styles", object()), "lag_trading_days", DEFAULT_LAG))
# except Exception:
#     pass
#
#
# # --------------------------- helpers ---------------------------
#
# def normalize_code(code: str) -> str:
#     """Normalize stock code to '######.(SH|SZ)' robustly."""
#     if pd.isna(code):
#         return code
#     s = str(code).upper().strip().replace("-", "").replace("_", "").replace(" ", "")
#     # patterns
#     if "." in s:
#         left, right = s.split(".", 1)
#         right = "SH" if right.startswith("SH") else ("SZ" if right.startswith("SZ") else right)
#         if left.isdigit() and len(left) == 6 and right in {"SH", "SZ"}:
#             return f"{left}.{right}"
#     if s.endswith("SHSE") and len(s) >= 10 and s[-8:-4].isdigit():
#         return f"{s[:-4]}.SH"
#     if s.endswith("SZSE") and len(s) >= 10 and s[-8:-4].isdigit():
#         return f"{s[:-4]}.SZ"
#     if s.startswith("SH") and len(s) >= 8 and s[2:8].isdigit():
#         return f"{s[2:8]}.SH"
#     if s.startswith("SZ") and len(s) >= 8 and s[2:8].isdigit():
#         return f"{s[2:8]}.SZ"
#     if s.isdigit() and len(s) == 6:
#         exch = "SH" if s.startswith("6") else "SZ"
#         return f"{s}.{exch}"
#     return s
#
#
# def choose_col(df: pd.DataFrame, cand_names: List[str]) -> str | None:
#     """Return the first existing column in df that matches candidate names (case-insensitive)."""
#     if df is None or df.empty:
#         return None
#     cols_lower = {c.lower(): c for c in df.columns}
#     for nm in cand_names:
#         c = cols_lower.get(nm.lower())
#         if c is not None:
#             return c
#     return None
#
#
# def coverage_ratio(s: pd.Series) -> float:
#     """Not-NaN coverage ratio over series length; returns 0 on empty."""
#     if s is None or len(s) == 0:
#         return 0.0
#     return float(np.mean(~pd.isna(s.values)))
#
#
# # --------------------------- diagnostics core ---------------------------
#
# def build_requirements() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
#     """
#     Returns:
#         base_synonyms: mapping canonical_base -> list of possible source column names
#         style_require: mapping style_name -> list of required canonical_base names
#     """
#     # Canonical base fields and their common synonyms from various vendor schemas.
#     base_synonyms = {
#         "market_cap":       ["market_cap", "mkt_cap", "total_mktcap", "tot_mv", "total_mv"],
#         "float_mktcap":     ["float_mktcap", "free_float_mktcap", "float_mv", "circ_mv"],
#         "book_value":       ["book_value", "book_equity", "net_assets", "shareholders_equity", "total_equity"],
#         "net_income_ttm":   ["net_income_ttm", "ni_ttm", "net_profit_ttm", "np_ttm"],
#         "revenue_ttm":      ["revenue_ttm", "sales_ttm", "oper_rev_ttm", "revenue_total_ttm"],
#         "ocf_ttm":          ["ocf_ttm", "oper_cf_ttm", "net_cf_oper_ttm", "cashflow_oper_ttm"],
#         "total_assets":     ["total_assets", "assets_total"],
#         "total_equity":     ["total_equity", "equity_total", "owners_equity"],
#         "roe_ttm":          ["roe_ttm", "roe_weighted_ttm", "roe_diluted_ttm"],
#         "roa_ttm":          ["roa_ttm", "roa_diluted_ttm"],
#         "growth_rev_yoy":   ["growth_rev_yoy", "revenue_yoy", "sales_yoy"],
#         "growth_np_yoy":    ["growth_np_yoy", "netprofit_yoy", "ni_yoy"],
#         # Optional share counts (for proxy diagnostics only; not used to compute coverage):
#         "total_shares":     ["total_shares", "shares_total"],
#         "float_shares":     ["float_shares", "free_float_shares", "tradable_shares"],
#         # As-of date columns for lag filtering diagnostics (optional):
#         "announce_date":    ["announce_date", "pub_date", "report_date", "asof_date"],
#         # Close price (only for potential proxy note):
#         "close":            ["close", "close_price"],
#     }
#
#     # Style factors → required canonical base fields
#     style_require = {
#         "size":         ["float_mktcap"],                      # proxy: market_cap if float_mktcap missing
#         "bp":           ["book_value", "market_cap"],          # book/price
#         "ep_ttm":       ["net_income_ttm", "market_cap"],
#         "sp_ttm":       ["revenue_ttm", "market_cap"],
#         "cf_yield":     ["ocf_ttm", "market_cap"],
#         "leverage":     ["total_assets", "total_equity"],
#         "roe_ttm":      ["roe_ttm"],
#         "roa_ttm":      ["roa_ttm"],
#         "growth_rev_yoy": ["growth_rev_yoy"],
#         "growth_np_yoy":  ["growth_np_yoy"],
#     }
#     return base_synonyms, style_require
#
#
# def diagnose(date: pd.Timestamp, lag_days: int, index_code: str, verbosity: str = "STEP") -> pd.DataFrame:
#     set_verbosity(verbosity)
#
#     step(f"[SETUP] target date = {date.date()}, lag_days = {lag_days}, index = {index_code}")
#     # 1) Get CSI500 members for the day (directly via API to avoid universe cache side-effects)
#     codes = get_index_members(index_code, date)
#     codes = sorted({normalize_code(c) for c in codes})
#     done(f"[SETUP] members = {len(codes)}")
#
#     if len(codes) == 0:
#         raise RuntimeError("No index members returned; abort diagnostics.")
#
#     # 2) Fetch fundamentals snapshot
#     step("[FETCH] fundamentals snapshot ...")
#     funda = get_fundamentals_snapshot(date, codes, lag_days=lag_days)
#     if funda is None or len(funda) == 0:
#         raise RuntimeError("Empty fundamentals snapshot; cannot proceed.")
#     # Normalize code column if present
#     code_col = choose_col(funda, ["code", "sec_code", "ticker"])
#     if code_col is None:
#         warn("[FETCH] No 'code' column found in snapshot; matching may fail.")
#         funda_codes = []
#     else:
#         funda[code_col] = funda[code_col].map(normalize_code)
#         funda_codes = funda[code_col].tolist()
#
#     matched = len(set(funda_codes).intersection(codes)) if funda_codes else 0
#     done(f"[FETCH] snapshot shape={funda.shape}, matched_codes={matched}/{len(codes)}")
#
#     # 3) Build synonym & requirement maps
#     base_synonyms, style_require = build_requirements()
#
#     # 4) Resolve actual columns & coverage for canonical base fields
#     step("[CHECK] base field coverage")
#     base_rows = []
#     resolved_cols: Dict[str, str | None] = {}
#     for canon, cands in base_synonyms.items():
#         real_col = choose_col(funda, cands)
#         resolved_cols[canon] = real_col
#         if real_col is None:
#             base_rows.append((canon, None, 0.0, "MISSING"))
#         else:
#             # coverage only over today's index members if code col exists; else overall
#             if code_col:
#                 sub = funda[funda[code_col].isin(codes)]
#                 cov = coverage_ratio(sub[real_col])
#             else:
#                 cov = coverage_ratio(funda[real_col])
#             base_rows.append((canon, real_col, cov, "OK" if cov > 0 else "ALL-NaN"))
#
#     base_df = pd.DataFrame(base_rows, columns=["canonical", "source_col", "coverage", "status"])
#     for _, row in base_df.sort_values("canonical").iterrows():
#         tag = row["status"]
#         cov = f"{row['coverage']:.2f}"
#         src = row["source_col"] or "-"
#         if tag == "MISSING":
#             warn(f"[BASE] {row['canonical']:<16} -> {tag}")
#         else:
#             done(f"[BASE] {row['canonical']:<16} -> {src:<24} coverage={cov}")
#
#     # 5) Infer style readiness based on required bases
#     step("[CHECK] style readiness (which styles are blocked)")
#     style_rows = []
#     THRESH = 0.60  # desired minimal coverage
#     for sty, req_bases in style_require.items():
#         req_missing = [b for b in req_bases if resolved_cols.get(b) is None]
#         # coverage is minimum over required bases (0 if any missing)
#         if len(req_missing) > 0:
#             style_rows.append((sty, 0.0, "BLOCKED: missing " + ",".join(req_missing)))
#             warn(f"[STYLE] {sty:<14} -> BLOCKED (missing: {','.join(req_missing)})")
#             continue
#
#         req_cov = []
#         for b in req_bases:
#             col = resolved_cols[b]
#             if code_col:
#                 sub = funda[funda[code_col].isin(codes)]
#                 req_cov.append(coverage_ratio(sub[col]))
#             else:
#                 req_cov.append(coverage_ratio(funda[col]))
#         cov_min = float(np.min(req_cov) if req_cov else 0.0)
#         status = "OK" if cov_min >= THRESH else ("WEAK(<0.6)" if cov_min > 0 else "ALL-NaN")
#         style_rows.append((sty, cov_min, status))
#         if status == "OK":
#             done(f"[STYLE] {sty:<14} -> coverage(min)={cov_min:.2f} OK")
#         elif status.startswith("WEAK"):
#             warn(f"[STYLE] {sty:<14} -> coverage(min)={cov_min:.2f} WEAK (<0.6)")
#         else:
#             warn(f"[STYLE] {sty:<14} -> coverage(min)={cov_min:.2f} ALL-NaN")
#
#     style_df = pd.DataFrame(style_rows, columns=["style", "min_required_coverage", "status"])
#
#     # 6) Friendly summary table (styles first)
#     step("[SUMMARY] styles (blocked / weak / ok)")
#     show = style_df.sort_values(["status", "style"]).copy()
#     # Highlight: BLOCKED first, then WEAK, then OK
#     status_order = {"BLOCKED": 0, "WEAK(<0.6)": 1, "OK": 2, "ALL-NaN": 0}
#     show["_ord"] = show["status"].map(lambda s: status_order.get(s.split()[0], 3))
#     show = show.sort_values(["_ord", "style"]).drop(columns=["_ord"])
#     # Print as plain text table
#     with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 120):
#         print("\n==== Style Readiness ====")
#         print(show.to_string(index=False, formatters={"min_required_coverage": "{:.2f}".format}))
#         print("\n==== Base Field Coverage ====")
#         print(base_df.sort_values(["status","canonical"]).to_string(index=False, formatters={"coverage": "{:.2f}".format}))
#
#     done("[SUMMARY] diagnostics completed.")
#     return style_df
#
#
# # --------------------------- CLI ---------------------------
#
# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Diagnose missing fields for style subfactors.")
#     parser.add_argument("--date", type=str, default=None, help="Target trading date YYYY-MM-DD (default: last trading day).")
#     parser.add_argument("--lag", type=int, default=DEFAULT_LAG, help=f"Lag trading days (default: {DEFAULT_LAG}).")
#     parser.add_argument("--index", type=str, default="SHSE.000905", help="Index code for universe (default: SHSE.000905 CSI500).")
#     parser.add_argument("--verbosity", type=str, default="STEP", help="Log verbosity: SILENT/STEP/LOOP/DEBUG.")
#     return parser.parse_args()
#
#
# def main() -> None:
#     args = parse_args()
#     set_verbosity(args.verbosity)
#
#     # Resolve target date: last trading day if not provided
#     if args.date is None:
#         step("[CAL] resolve last trading day ...")
#         # use last 30 calendar days as a safe window
#         days = get_trade_days(pd.Timestamp.today().date().strftime("%Y-%m-%d"), pd.Timestamp.today().date().strftime("%Y-%m-%d"))
#         # Fallback if API expects a range: try last 30 days
#         if not days:
#             thirty = (pd.Timestamp.today() - pd.Timedelta(days=30)).date().strftime("%Y-%m-%d")
#             today = pd.Timestamp.today().date().strftime("%Y-%m-%d")
#             days = get_trade_days(thirty, today)
#         if not days:
#             raise RuntimeError("Unable to resolve trading days via API.")
#         target_date = pd.Timestamp(days[-1])
#         done(f"[CAL] last trading day = {target_date.date()}")
#     else:
#         target_date = pd.Timestamp(args.date)
#
#     # Run diagnostics
#     _ = diagnose(target_date, args.lag, args.index, args.verbosity)
#
#
# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as ex:
#         error(f"[FATAL] {ex}")
#         raise

# file: test_fetch_styles_fields_20240102.py
"""
Test purpose
------------
Precisely fetch and verify fundamental fields required by style subfactors on
the SINGLE trading day 2024-01-02 for CSI500, with lag filtering applied
upstream by myquant_io.get_fundamentals_snapshot. It reports:
- Which canonical fields exist (with resolved source column);
- Coverage ratio among index members (non-NaN share);
- Code normalization/matching rate (avoid SHSE.600000 vs 600000.SH mismatch).

This script does NOT modify pipeline outputs. It's read-only diagnostics.

Run
---
python test_fetch_styles_fields_20240102.py --lag 5 --index SHSE.000905 --verbosity STEP
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.utils import logging as log  # type: ignore
from src.api.myquant_io import (  # type: ignore
    get_index_members,
    get_fundamentals_snapshot,
)

# >>> 新增：直接使用 GM 公版 SDK 做“测试用补源” <<<
from gm.api import stk_get_daily_basic_pt, stk_get_fundamentals_balance_pt  # type: ignore

TARGET_DATE = pd.Timestamp("2024-01-02")
DEFAULT_INDEX = "SHSE.000905"  # CSI500

def _norm_code(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return x
    s = str(x).strip().upper().replace("_", ".").replace("-", "")
    if s.startswith("SHSE.") or s.startswith("SZSE."):
        mk, num = s.split(".", 1)
        suf = "SH" if mk.startswith("SH") else "SZ"
        return f"{num.zfill(6)}.{suf}"
    if s.startswith("SH") and len(s) >= 8 and s[2:8].isdigit():
        return f"{s[2:8]}.SH"
    if s.startswith("SZ") and len(s) >= 8 and s[2:8].isdigit():
        return f"{s[2:8]}.SZ"
    if "." in s:
        a, b = s.split(".", 1)
        b = "SH" if b.startswith("SH") else ("SZ" if b.startswith("SZ") else b)
        if a.isdigit() and len(a) == 6 and b in {"SH", "SZ"}:
            return f"{a}.{b}"
    if s.isdigit() and len(s) == 6:
        return f"{s}.SH" if s[0] == "6" else f"{s}.SZ"
    return s

def _to_gm_symbol(code: str) -> str:
    code = _norm_code(code)
    num, ex = code.split(".")
    pref = "SHSE" if ex == "SH" else "SZSE"
    return f"{pref}.{num}"

def _to_gm_symbols(codes: List[str]) -> List[str]:
    return [_to_gm_symbol(c) for c in codes]

def _choose_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower = {c.lower(): c for c in df.columns}
    for nm in cands:
        c = lower.get(nm.lower())
        if c:
            return c
    return None

def _coverage_xs(df: pd.DataFrame, code_col: str, codes: List[str], col: str) -> float:
    sub = df[df[code_col].isin(codes)]
    if sub.empty or col not in sub.columns:
        return 0.0
    return float(np.mean(~pd.isna(sub[col].values)))

# 别名表：把 GM 的键也加进去（ttl_* / circ_shr / ttl_shr / pub_date）
FUNDA_ALIAS: Dict[str, List[str]] = {
    "market_cap":   ["market_cap", "mkt_cap", "total_mktcap", "tot_mv", "total_mv", "总市值"],
    "float_mktcap": ["float_mktcap", "free_float_mktcap", "float_mv", "circ_mv", "流通市值"],
    "total_shares": ["total_shares", "ttl_shr", "shares_total", "总股本"],
    "float_shares": ["float_shares", "circ_shr", "ttl_shr_unl", "tradable_shares", "流通股本"],

    "pb":           ["pb", "pb_mrq", "pb_ttm", "市净率"],
    "pe_ttm":       ["pe_ttm", "市盈率TTM"],
    "ps_ttm":       ["ps_ttm", "市销率TTM"],
    "pcf_ttm":      ["pcf_ttm", "市现率TTM"],

    "roe_ttm":      ["roe_ttm", "roe_weighted_ttm", "净资产收益率TTM"],
    "roa_ttm":      ["roa_ttm", "roa_diluted_ttm", "总资产回报率TTM"],

    "net_profit_ttm": ["net_profit_ttm", "net_income_ttm", "ni_ttm", "np_ttm", "净利润TTM"],
    "revenue_ttm":    ["revenue_ttm", "sales_ttm", "oper_rev_ttm", "营业收入TTM"],
    "oper_cf_ttm":    ["oper_cf_ttm", "ocf_ttm", "net_cf_oper_ttm", "经营现金流TTM"],

    "growth_np_yoy":  ["growth_np_yoy", "netprofit_yoy", "ni_yoy", "净利润同比"],
    "growth_rev_yoy": ["growth_rev_yoy", "revenue_yoy", "sales_yoy", "营业收入同比"],

    # 资产负债表：加入 ttl_* 和组合后的 long_debt
    "total_assets": ["total_assets", "ttl_ast", "assets_total", "总资产", "资产总计"],
    "total_equity": ["total_equity", "ttl_eqy_pcom", "ttl_eqy", "owners_equity", "股东权益合计"],
    "total_debt":   ["total_debt", "ttl_liab", "liab_total", "负债合计", "总负债"],
    "long_debt":    ["long_debt", "bnd_pay+lt_ln+lt_pay+leas_liab"],

    "announce_date": ["announce_date", "pub_date", "公告日期", "发布日期"],

    "book_value":   ["book_value", "book_equity", "net_assets", "净资产", "归母净资产"],
}

def _augment_with_daily_and_balance(funda: pd.DataFrame, codes: List[str], lag_days: int) -> pd.DataFrame:
    """用 GM 的 pt 接口补齐股本 + 资产负债表，并标准化为 canonical 列名。"""
    gm_codes = _to_gm_symbols(codes)
    trade_date = TARGET_DATE.strftime("%Y-%m-%d")
    query_date = (TARGET_DATE - pd.Timedelta(days=lag_days)).strftime("%Y-%m-%d")

    # daily_basic：股本
    try:
        log.step(f"[FETCH][GM] daily_basic_pt symbols={len(gm_codes)} trade_date={trade_date}")
        db = stk_get_daily_basic_pt(symbols=gm_codes, fields="ttl_shr,circ_shr", trade_date=trade_date, df=True)
        if db is None or len(db) == 0:
            log.warn("[FETCH][GM] daily_basic_pt returned empty.")
            db = pd.DataFrame(columns=["symbol", "ttl_shr", "circ_shr"])
        db = db.copy()
        db["code"] = db["symbol"].map(_norm_code)
        db = db[["code", "ttl_shr", "circ_shr"]].drop_duplicates("code")
        db = db.rename(columns={"ttl_shr": "total_shares", "circ_shr": "float_shares"})
        log.done(f"[FETCH][GM] daily_basic rows={len(db)}")
    except Exception as e:
        log.warn(f"[FETCH][GM] daily_basic_pt failed: {e}")
        db = pd.DataFrame(columns=["code", "total_shares", "float_shares"])

    # balance：最小字段 + 组合长债、总权益优先归母
    bal_fields = ["ttl_ast", "ttl_liab", "ttl_eqy_pcom", "ttl_eqy", "bnd_pay", "lt_ln", "lt_pay", "leas_liab"]
    try:
        log.step(f"[FETCH][GM] balance_pt symbols={len(gm_codes)} date={query_date}")
        bal = stk_get_fundamentals_balance_pt(symbols=gm_codes, fields=",".join(bal_fields), date=query_date, df=True)
        if bal is None or len(bal) == 0:
            log.warn("[FETCH][GM] balance_pt returned empty.")
            bal = pd.DataFrame(columns=["symbol"] + bal_fields)
        bal = bal.copy()
        bal["code"] = bal["symbol"].map(_norm_code)
        # 组合列
        nums = bal[["bnd_pay", "lt_ln", "lt_pay", "leas_liab"]].apply(pd.to_numeric, errors="coerce")
        bal["long_debt"] = nums.sum(axis=1, min_count=1)  # 全 NaN 则 NaN
        bal["total_equity"] = pd.to_numeric(bal.get("ttl_eqy_pcom"), errors="coerce").combine_first(
            pd.to_numeric(bal.get("ttl_eqy"), errors="coerce")
        )
        bal = bal.rename(columns={
            "ttl_ast": "total_assets",
            "ttl_liab": "total_debt",
            "pub_date": "announce_date",
        })
        keep = ["code", "announce_date", "total_assets", "total_debt", "total_equity", "long_debt"]
        bal = bal[keep].drop_duplicates("code")
        log.done(f"[FETCH][GM] balance rows={len(bal)}")
    except Exception as e:
        log.warn(f"[FETCH][GM] balance_pt failed: {e}")
        bal = pd.DataFrame(columns=["code", "announce_date", "total_assets", "total_debt", "total_equity", "long_debt"])

    # 合并到 snapshot
    code_col = _choose_col(funda, ["code", "sec_code", "ticker", "symbol", "证券代码"])
    if code_col is None:
        f = funda.copy()
        f.index = pd.Index([_norm_code(x) for x in f.index], name="code")
        f = f.reset_index()
        code_col = "code"
    else:
        f = funda.copy()
        f[code_col] = f[code_col].astype(str).map(_norm_code)

    f = f.rename(columns={code_col: "code"})
    f = f.merge(db, on="code", how="left").merge(bal, on="code", how="left")
    return f

def diagnose_fields(lag_days: int, index_code: str, verbosity: str = "STEP") -> pd.DataFrame:
    log.set_verbosity(verbosity)
    log.step(f"[SETUP] date={TARGET_DATE.date()}, lag={lag_days}, index={index_code}")
    codes_raw = get_index_members(index_code, TARGET_DATE)
    codes = sorted({_norm_code(c) for c in codes_raw})
    log.done(f"[SETUP] members={len(codes)}")
    if len(codes) == 0:
        raise RuntimeError("Empty index members.")

    log.step("[FETCH] fundamentals snapshot (lag-filtered upstream)")
    funda_snap = get_fundamentals_snapshot(TARGET_DATE, codes, lag_days=lag_days)
    if funda_snap is None or len(funda_snap) == 0:
        raise RuntimeError("Empty fundamentals snapshot.")

    # >>> 新增：用 GM pt 接口补齐“股本 + 资产负债表”
    funda = _augment_with_daily_and_balance(funda_snap, codes, lag_days)

    matched = float(pd.Series(funda["code"]).isin(codes).mean())
    log.done(f"[DONE] Snapshot+Augmented fields={len(funda.columns)} cols, names={len(funda)}, matched={matched:.2f}")

    rows: List[Tuple[str, Optional[str], float, str]] = []
    for canon, aliases in FUNDA_ALIAS.items():
        # 确保把 canonical 名称自己也放到候选（已在 FUNDA_ALIAS 里）
        src = _choose_col(funda, aliases)
        if src is None:
            rows.append((canon, None, 0.0, "MISSING"))
            log.warn(f"[FIELD] {canon:<15} -> MISSING (aliases={aliases[:3]}...)")
            continue
        cov = _coverage_xs(funda, "code", codes, src)
        status = "OK" if cov > 0 else "ALL-NaN"
        rows.append((canon, src, cov, status))
        log.done(f"[FIELD] {canon:<15} -> {src:<20} coverage={cov:.2f}")

    df = pd.DataFrame(rows, columns=["canonical", "source_col", "coverage", "status"])

    log.step("[SUMMARY] Missing / All-NaN fields")
    miss = df[df["status"] == "MISSING"]
    allnan = df[df["status"] == "ALL-NaN"]
    if not miss.empty:
        print("\n-- MISSING --")
        print(miss.sort_values("canonical").to_string(index=False, formatters={"coverage": "{:.2f}".format}))
    if not allnan.empty:
        print("\n-- ALL-NaN --")
        print(allnan.sort_values("canonical").to_string(index=False, formatters={"coverage": "{:.2f}".format}))

    log.step("[SUMMARY] Top-20 columns (quick glance)")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        cols_preview = list(funda.columns)[:20]
        print(f"\ncolumns[:20] = {cols_preview}")

    sample = funda[funda["code"].isin(codes)].head(3)
    print("\n-- SAMPLE (first 3 matched rows) --")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(sample.to_string(index=False))

    log.done("[DONE] diagnostics completed.")
    return df

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch & verify style-required fields on 2024-01-02.")
    p.add_argument("--lag", type=int, default=5, help="Lag (trading days) for fundamentals snapshot; default=5")
    p.add_argument("--index", type=str, default=DEFAULT_INDEX, help="Index code for universe; default=SHSE.000905")
    p.add_argument("--verbosity", type=str, default="STEP", help="Log verbosity: SILENT/STEP/LOOP/DEBUG")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    diagnose_fields(lag_days=args.lag, index_code=args.index, verbosity=args.verbosity)

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log.error(f"[FATAL] {ex}")
        raise


# import pandas as pd
# import src.api.myquant_io as myquant_io  # 避免新增契约
# t = pd.Timestamp("2024-01-02")
# idx = "SHSE.000905"
# lag = 5
#
#
# codes = myquant_io.get_index_members(index_code=idx, date=t)
#
#
#
# from gm.api import stk_get_fundamentals_balance_pt
#
#
#
#     # —— 只要我们计算 long_debt/book_equity 会用到的最小字段（<=20）——
# bal_fields = [
#         "ttl_ast",       # 资产总计
#         "ttl_liab",      # 负债合计
#         "ttl_eqy_pcom",  # 归母权益
#         "ttl_eqy",       # 股东权益合计（兜底用）
#         "bnd_pay",       # 应付债券
#         "lt_ln",         # 长期借款
#         "lt_pay",        # 长期应付款
#         "leas_liab",     # 租赁负债
#     ]
# fields_str = ",".join(bal_fields)
# query_date = (t - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
#
# df = None
#
#
# df = stk_get_fundamentals_balance_pt(
#             symbols=list(codes),
#             fields=fields_str,
#             date=query_date,
#             df=True,
#         )
#
# print(df)