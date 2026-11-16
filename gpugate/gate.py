"""
gate.py: a GPU traffic cop for background jobs.

Problem: one consumer GPU (say 16 GB) is shared by background jobs (local LLMs,
image generation) and programs a person runs themselves (games, music tools,
browsers). When they don't fit together, something gets pushed out of video
memory (slow) or crashes.

Approach:
  1. Read how much video memory each program holds on each card.
  2. Split programs into three groups: *ours* (the jobs we manage), *ignored*
     (the OS and everyday apps, which shrink when someone needs room), and
     *others* (everything else).
  3. Before a job loads a model, ask the gate whether it fits beside the others.
     If not, the job waits and says who is in the way. It never closes,
     pauses or slows anyone else's program.
  4. A wait has a time limit. A wrong reading must not stall a whole queue.
  5. If the reading is missing or stale, the gate lets the job go ahead: it
     can't see, so it doesn't guess.

The card's own total is treated as the truth. Per-program readings overlap
(shared memory is counted once per program), so they are never added up past
what the card reports.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

FRESH_SECONDS = 90          # a reading older than this is ignored
MARGIN_GB = 1.0             # headroom kept free on top of what everyone uses
NAME_FROM_GB = 1.0          # programs holding at least this much are named in messages

_PROC_RE = re.compile(r"gpu process memory\(pid_(\d+)_(luid_0x[0-9a-f]+_0x[0-9a-f]+)")
_CARD_RE = re.compile(r"gpu adapter memory\((luid_0x[0-9a-f]+_0x[0-9a-f]+)")

_POWERSHELL = r"""
$ErrorActionPreference = 'SilentlyContinue'
$c = Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Dedicated Usage'
if ($c) { $c.CounterSamples | Select-Object Path, CookedValue | ConvertTo-Json -Compress } else { '[]' }
"""


@dataclass
class Card:
    """One graphics card as reported in a single reading."""
    luid: str
    used_gb: float = 0.0
    procs: dict[int, float] = field(default_factory=dict)  # pid -> GB held on this card


@dataclass
class Picture:
    """The main card split into ours, others, and ignored programs."""
    total_gb: float
    used_gb: float
    ours_gb: float
    others: list[tuple[str, float]]
    others_gb: float

    @property
    def free_for_jobs(self) -> float:
        # Never trust per-program numbers past the card's total minus our share.
        others = min(self.others_gb, max(0.0, self.used_gb - self.ours_gb))
        return self.total_gb - others - MARGIN_GB


def parse_windows_counters(samples: Iterable[dict]) -> dict[str, Card]:
    """Windows performance-counter samples -> one Card per GPU.

    Pure function, so tests can feed it made-up readings. A program can have
    several entries on one card; they are added up. A card with no adapter
    total has no usable reading and is dropped.
    """
    cards: dict[str, Card] = {}
    for s in samples or []:
        path = (s.get("Path") or "").lower()
        gb = float(s.get("CookedValue") or 0) / 1024**3
        if m := _PROC_RE.search(path):
            pid, luid = int(m.group(1)), m.group(2)
            card = cards.setdefault(luid, Card(luid))
            card.procs[pid] = card.procs.get(pid, 0.0) + gb
        elif m := _CARD_RE.search(path):
            card = cards.setdefault(m.group(1), Card(m.group(1)))
            card.used_gb = max(card.used_gb, gb)
    return {luid: c for luid, c in cards.items() if c.used_gb > 0}


def read_windows() -> list[dict] | None:
    """One PowerShell call for per-program and per-card GPU memory. None if unavailable."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _POWERSHELL],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip()
        data = json.loads(out or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return [data] if isinstance(data, dict) else data


class Gate:
    """Decides whether a job fits on the GPU beside everyone else.

    total_gb       memory on the main card
    is_ours(pid)   True for programs the gate manages (e.g. the model server)
    classify(pid)  returns (program_name, display_name) for any other program
    ignore         program names (lowercase) the gate never waits for
    reader         returns Windows-style counter samples (swap it to test)
    """

    def __init__(
        self,
        total_gb: float,
        is_ours: Callable[[int], bool] = lambda pid: False,
        classify: Callable[[int], tuple[str, str]] = lambda pid: (f"p{pid}", f"program {pid}"),
        ignore: Iterable[str] = (),
        reader: Callable[[], list[dict] | None] = read_windows,
        clock: Callable[[], float] = time.time,
    ):
        self.total_gb = total_gb
        self._is_ours = is_ours
        self._classify = classify
        self._ignore = {n.lower() for n in ignore}
        self._reader = reader
        self._clock = clock
        self._lock = threading.Lock()
        self._cards: dict[str, Card] = {}
        self._main: str | None = None
        self._seen_at = 0.0

    # --- reading -----------------------------------------------------------------

    def look(self, samples: list[dict] | None = None) -> bool:
        """Take one reading. True if the reader answered."""
        if samples is None:
            samples = self._reader()
        if samples is None:
            return False
        cards = parse_windows_counters(samples)
        with self._lock:
            self._cards = cards
            self._pick_main(cards)
            self._seen_at = self._clock()
        return True

    def _pick_main(self, cards: dict[str, Card]) -> None:
        """The card with the most memory in use. It sticks, so a brief spike on a
        small integrated GPU doesn't swap cards; another card must pass it by 2 GB."""
        if not cards:
            self._main = None
            return
        best = max(cards.values(), key=lambda c: c.used_gb)
        current = cards.get(self._main or "")
        if current is None or best.used_gb > current.used_gb + 2:
            self._main = best.luid

    def _main_card(self) -> Card | None:
        with self._lock:
            if self._clock() - self._seen_at > FRESH_SECONDS:
                return None
            return self._cards.get(self._main or "")

    def picture(self) -> Picture | None:
        card = self._main_card()
        if card is None:
            return None
        ours = sum(gb for pid, gb in card.procs.items() if self._is_ours(pid))
        named: list[tuple[str, float]] = []
        others = 0.0
        for pid, gb in sorted(card.procs.items(), key=lambda kv: -kv[1]):
            if self._is_ours(pid) or gb < 0.1:
                continue
            program, shown = self._classify(pid)
            if program.lower() in self._ignore:
                continue
            others += gb
            if gb >= NAME_FROM_GB:
                named.append((shown, gb))
        return Picture(self.total_gb, card.used_gb, ours, named, others)

    # --- the rules ---------------------------------------------------------------

    def room_for(self, needs_gb: float) -> tuple[bool, str]:
        """(fits, why). With no fresh reading it says yes: it can't see, so it doesn't guess."""
        pic = self.picture()
        if pic is None or pic.free_for_jobs >= needs_gb:
            return True, ""
        if pic.others:
            who = ", ".join(f"{n} is using {g:.1f} GB" for n, g in pic.others[:2])
        else:
            who = f"other programs are using {pic.others_gb:.1f} GB"
        return False, f"{who}, so there is only {max(0.0, pic.free_for_jobs):.1f} GB left"

    def wait_for_room(
        self,
        needs_gb: float,
        on_wait: Callable[[str], None] | None = None,
        max_wait_s: float = 20 * 60,
        poll_s: float = 20,
        cancelled: Callable[[], bool] = lambda: False,
        sleep: Callable[[float], None] = time.sleep,
        refresh: bool = True,
    ) -> str:
        """Hold a job until it fits. Returns "room", "timeout" or "cancelled".

        on_wait(msg) is called when waiting starts and whenever who is in the
        way changes, so a job's status stays true. After max_wait_s the job goes
        ahead anyway: a wrong reading must never stall a queue.
        """
        deadline = self._clock() + max_wait_s
        said = ""
        while True:
            if cancelled():
                return "cancelled"
            if refresh:
                self.look()
            fits, why = self.room_for(needs_gb)
            if fits:
                return "room"
            if self._clock() >= deadline:
                if on_wait:
                    on_wait(f"Waited {max_wait_s / 60:g} minutes ({why}). Going ahead anyway.")
                return "timeout"
            who = re.sub(r"[\d.]+ GB", "", why.split(", so")[0])
            if on_wait and who != said:
                on_wait(f"Waiting for GPU room: {why}.")
            said = who
            sleep(poll_s)

    def crowded(self, others_at_start: float | None) -> str:
        """While a job runs: why it should stop, or "". Stops when other programs grew
        by 2 GB or more since the job started and the card is nearly full."""
        pic = self.picture()
        if pic is None or others_at_start is None:
            return ""
        if pic.others_gb - others_at_start >= 2 and pic.used_gb >= pic.total_gb - MARGIN_GB:
            return ", ".join(f"{n} {g:.1f} GB" for n, g in pic.others[:2]) or "other programs"
        return ""
