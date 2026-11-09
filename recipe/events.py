"""A tiny event bus. Everything interesting emits an event; sinks decide what to do with it.

Default sink: append JSON lines to runs/live/events.jsonl. Add sinks with add_sink(fn). Sinks must never raise.
Set RECIPE_EVENTS=0 to turn it off.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

_lock = threading.Lock()
_sinks: list[Callable[[dict], None]] = []


def runs_root() -> Path:
    return Path(os.environ.get("RECIPE_RUNS", Path(__file__).resolve().parent.parent / "runs"))


def _file_sink(ev: dict) -> None:
    p = runs_root() / "live" / "events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock, p.open("a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def add_sink(fn: Callable[[dict], None]) -> None:
    _sinks.append(fn)


def emit(kind: str, **fields) -> None:
    if os.environ.get("RECIPE_EVENTS", "1") == "0":
        return
    ev = {"t": round(time.time(), 3), "kind": kind, **fields}
    for sink in [_file_sink, *_sinks]:
        try:
            sink(ev)
        except Exception:
            pass
