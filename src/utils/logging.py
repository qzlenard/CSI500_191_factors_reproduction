"""
[CN] 日志工具契约：统一格式、级别与落盘（控制台 + 文件）。
[Purpose] Structured progress logging.

Interfaces:
- set_verbosity(level: str) -> None                 # "SILENT"/"STEP"/"LOOP"/"DEBUG"
- bind_logfile(path: str) -> None                   # also mirror to file
- step(msg: str) -> None                            # [STEP] ...
- loop_progress(msg: str, done: int, total: int, eta: str|None=None) -> None  # [LOOP] ...
- done(msg: str) -> None                            # ... done
- warn(msg: str, exc: Exception|None=None) -> None  # [WARN]
- error(msg: str, exc: Exception|None=None) -> None # [ERROR]

Format Examples:
- "[STEP 1/6] Fetch OHLCV ..."; "[REG] R2=0.18, N=432"; "[ALPHA] saved 468 names"
"""
from __future__ import annotations
# file: src/utils/logging.py
# -*- coding: utf-8 -*-
"""
Lightweight project logger for the CSI500-191 pipeline.

Features
--------
- Dual sink: console + rotating log file under out/logs/
- Verbosity levels: SILENT(0)/STEP(1)/LOOP(2)/DEBUG(3)
- Step logs, loop progress with ETA, completion marker
- Tagged metrics: e.g., metric("REG", R2=0.18, N=432)
- Thread-safe writes; ANSI color in TTY (auto fallback)
- Minimal dependencies (stdlib only)

Public API (stable)
-------------------
set_verbosity(level: int | str) -> None
bind_logfile(log_dir: str="out/logs", prefix: str="run", rotate_daily: bool=True) -> str
step(msg: str, i: int | None=None, n: int | None=None) -> None
loop_progress(task: str, current: int, total: int, *, start_time: float | None=None,
              every: int=1, extra: dict | None=None) -> float
done(msg: str="done") -> None
warn(msg: str, exc: BaseException | None=None) -> None
error(msg: str, exc: BaseException | None=None) -> None
debug(msg: str, extra: dict | None=None) -> None
metric(tag: str, **fields) -> None
"""


import os
import sys
import json
import time
import traceback
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Union, IO

# IMPORTANT:
# Use a distinct alias for stdlib math to avoid shadowing by our own src/utils/numerics.py
import math as _pymath

# -------- Verbosity ---------------------------------------------------------

_VERBOSITY_NAME_TO_INT = {
    "SILENT": 0,
    "STEP": 1,
    "LOOP": 2,
    "DEBUG": 3,
    "0": 0, "1": 1, "2": 2, "3": 3,
}

_INT_TO_VERBOSITY_NAME = {v: k for k, v in _VERBOSITY_NAME_TO_INT.items() if isinstance(v, int)}

# -------- Internal singleton ------------------------------------------------

class _ProjectLogger:
    def __init__(self) -> None:
        self._verbosity: int = 1  # default STEP
        self._fh: Optional[IO[str]] = None   # file handle with write/flush/close
        self._log_path: Optional[str] = None
        self._lock = threading.RLock()
        self._enable_color = getattr(sys.stderr, "isatty", lambda: False)()
        self._session_start = time.time()

    # ----- configuration -----

    def set_verbosity(self, level: Union[int, str]) -> None:
        with self._lock:
            if isinstance(level, str):
                level = level.strip().upper()
                if level not in _VERBOSITY_NAME_TO_INT:
                    raise ValueError(f"Unknown verbosity: {level}")
                self._verbosity = _VERBOSITY_NAME_TO_INT[level]
            elif isinstance(level, int):
                if level not in (0, 1, 2, 3):
                    raise ValueError("verbosity must be 0..3")
                self._verbosity = level
            else:
                raise TypeError("verbosity must be int or str")

    def bind_logfile(self, log_dir: str = "out/logs", prefix: str = "run",
                     rotate_daily: bool = True) -> str:
        """
        Create log directory and open a new file.
        Returns the log file path.
        """
        with self._lock:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{ts}_{prefix}.log" if rotate_daily else f"{prefix}.log"
            path = os.path.join(log_dir, fname)
            # Close previous handle if any
            if self._fh is not None:
                try:
                    self._fh.flush()
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None
            self._fh = open(path, "a", encoding="utf-8", buffering=1)
            self._log_path = path

            # Write a header
            hdr = {
                "event": "LOG_START",
                "time": self._now_iso(),
                "verbosity": _INT_TO_VERBOSITY_NAME.get(self._verbosity, str(self._verbosity)),
                "pid": os.getpid(),
                "cwd": os.getcwd(),
                "session_uptime_sec": 0.0,
            }
            self._emit("INFO", "Log session started", structured=hdr)
            return path

    # ----- basic writers -----

    def step(self, msg: str, i: Optional[int] = None, n: Optional[int] = None) -> None:
        if self._verbosity < 1:
            return
        prefix = f"[STEP {i}/{n}]" if (i is not None and n is not None) else "[STEP]"
        self._emit("STEP", f"{prefix} {msg}")

    def loop_progress(self, task: str, current: int, total: int, *,
                      start_time: Optional[float] = None, every: int = 1,
                      extra: Optional[Dict[str, Any]] = None) -> float:
        """
        Print progress for long loops with ETA. Returns start_time for convenience.
        Only prints when (current % every == 0) or current == total.
        """
        if self._verbosity < 2:
            return start_time if start_time is not None else time.time()

        if total <= 0:
            total = max(total, 1)

        should_print = (current == total) or (every <= 1) or (current % max(every, 1) == 0)
        if not should_print:
            return start_time if start_time is not None else time.time()

        now = time.time()
        st = start_time if start_time is not None else now
        elapsed = max(now - st, 1e-9)
        rate = current / elapsed if current > 0 else 0.0
        remain = (total - current) / rate if rate > 0 else float("inf")
        eta = self._fmt_hms(remain)
        pct = (current / total) * 100.0
        msg = f"[LOOP] {task} ({current}/{total}, {pct:.1f}%) ETA ~ {eta}"
        payload = {"event": "LOOP", "task": task, "current": current, "total": total,
                   "pct": pct, "eta_sec": remain, "elapsed_sec": elapsed, "rate_per_sec": rate}
        if extra:
            payload.update(extra)
        self._emit("LOOP", msg, structured=payload)
        return st

    def done(self, msg: str = "done") -> None:
        if self._verbosity < 1:
            return
        self._emit("DONE", f"[DONE] {msg}")

    def warn(self, msg: str, exc: Optional[BaseException] = None) -> None:
        if self._verbosity < 1:
            return
        payload = {"event": "WARN"}
        if exc is not None:
            payload["exc_type"] = type(exc).__name__
            payload["exc"] = self._exc_to_str(exc)
        self._emit("WARN", f"[WARN] {msg}", structured=payload, stream="stderr")

    def error(self, msg: str, exc: Optional[BaseException] = None) -> None:
        payload = {"event": "ERROR"}
        if exc is not None:
            payload["exc_type"] = type(exc).__name__
            payload["exc"] = self._exc_to_str(exc)
        self._emit("ERROR", f"[ERROR] {msg}", structured=payload, stream="stderr")

    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if self._verbosity < 3:
            return
        payload = {"event": "DEBUG"}
        if extra:
            payload.update(extra)
        self._emit("DEBUG", f"[DEBUG] {msg}", structured=payload)

    def metric(self, tag: str, **fields: Any) -> None:
        """
        Log a tagged metric line. Example:
        >>> metric("REG", R2=0.18, N=432)
        Emits: [REG] R2=0.18, N=432
        """
        if self._verbosity < 1:
            return
        text = ", ".join(f"{k}={fields[k]}" for k in sorted(fields.keys()))
        msg = f"[{tag}] {text}" if text else f"[{tag}]"
        payload = {"event": "METRIC", "tag": tag, **fields}
        self._emit("METRIC", msg, structured=payload)

    # ----- low-level emit -----

    def _emit(self, level: str, text: str, *, structured: Optional[Dict[str, Any]] = None,
              stream: str = "stderr") -> None:
        ts = self._now_iso()
        human = f"{ts} {text}"
        colored = self._colorize(level, human) if self._enable_color else human

        with self._lock:
            # Console
            fd = sys.stderr if stream == "stderr" else sys.stdout
            try:
                fd.write(colored + "\n")
                fd.flush()
            except Exception:
                pass

            # File
            if self._fh is not None:
                try:
                    self._fh.write(human + "\n")
                    if structured:
                        record = {"time": ts, "level": level, **structured}
                        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    self._fh.flush()
                except Exception:
                    # Avoid crashing the pipeline due to logging failures
                    pass

    # ----- helpers -----

    @staticmethod
    def _fmt_hms(sec: float) -> str:
        if _pymath.isinf(sec) or sec >= 86400 * 100:
            return "∞"
        sec = max(0.0, sec)
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _exc_to_str(exc: BaseException) -> str:
        try:
            return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:
            return repr(exc)

    @staticmethod
    def _now_iso() -> str:
        # Local time in ISO format with seconds precision
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _colorize(level: str, msg: str) -> str:
        # Minimal ANSI color map
        C = {
            "STEP": "\033[1;36m",   # bold cyan
            "LOOP": "\033[0;34m",   # blue
            "DONE": "\033[0;32m",   # green
            "WARN": "\033[0;33m",   # yellow
            "ERROR": "\033[0;31m",  # red
            "DEBUG": "\033[0;37m",  # gray
            "INFO": "\033[0;36m",   # cyan
            "METRIC": "\033[1;35m", # magenta
        }
        R = "\033[0m"
        color = C.get(level, "")
        return f"{color}{msg}{R}" if color else msg

# -------- module-level facade ----------------------------------------------

_LOGGER = _ProjectLogger()

def set_verbosity(level: Union[int, str]) -> None:
    """Set global verbosity. One of {0..3, 'SILENT','STEP','LOOP','DEBUG'}."""
    _LOGGER.set_verbosity(level)

def bind_logfile(log_dir: str = "out/logs", prefix: str = "run",
                 rotate_daily: bool = True) -> str:
    """
    Bind a new logfile under `log_dir`. Returns the logfile path.
    Typical call at program start. Safe to call multiple times.
    """
    return _LOGGER.bind_logfile(log_dir=log_dir, prefix=prefix, rotate_daily=rotate_daily)

def step(msg: str, i: Optional[int] = None, n: Optional[int] = None) -> None:
    """Print a STEP-level message. Optional i/n for '[STEP i/n]' prefix."""
    _LOGGER.step(msg, i=i, n=n)

def loop_progress(task: str, current: int, total: int, *,
                  start_time: Optional[float] = None, every: int = 1,
                  extra: Optional[Dict[str, Any]] = None) -> float:
    """
    Print LOOP-level progress with ETA. Returns `start_time` (unchanged or now if None).
    Use pattern:
        t0 = time.time()
        for k in range(1, N+1):
            # do work...
            t0 = loop_progress("Task", k, N, start_time=t0, every=10)
    """
    return _LOGGER.loop_progress(task, current, total, start_time=start_time, every=every, extra=extra)

def done(msg: str = "done") -> None:
    """Print a DONE marker (STEP-level)."""
    _LOGGER.done(msg)

def warn(msg: str, exc: Optional[BaseException] = None) -> None:
    """Print a WARN message (STEP-level). Optionally attach exception stack."""
    _LOGGER.warn(msg, exc=exc)

def error(msg: str, exc: Optional[BaseException] = None) -> None:
    """Print an ERROR message (always logged). Optionally attach exception stack."""
    _LOGGER.error(msg, exc=exc)

def debug(msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Print a DEBUG message if verbosity >= DEBUG."""
    _LOGGER.debug(msg, extra=extra)

def metric(tag: str, **fields: Any) -> None:
    """Print a tagged metric line, e.g., metric('REG', R2=0.18, N=432)."""
    _LOGGER.metric(tag, **fields)

# -------- graceful shutdown -------------------------------------------------

import atexit

def _on_exit() -> None:
    try:
        _LOGGER._emit("INFO", "Log session closed", structured={
            "event": "LOG_END",
            "time": _LOGGER._now_iso(),
        })
        if _LOGGER._fh is not None:
            _LOGGER._fh.flush()
            _LOGGER._fh.close()
            _LOGGER._fh = None
    except Exception:
        pass

atexit.register(_on_exit)
