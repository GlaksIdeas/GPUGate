# gpugate

A small GPU traffic cop for background jobs. When local LLMs, image generators
and the programs you run yourself (games, music tools, browsers) share one
consumer GPU, something gets pushed out of video memory or crashes. `gpugate`
lets a job check for room first, wait politely, and say who is in the way.

I built the original inside a personal AI agent that runs local models on a
16 GB card while I play games on the same PC. This is that idea as a
standalone, tested library with no dependencies.

## The rules

1. **Read** video memory per program, per card (Windows performance counters).
2. **Split** programs into *ours* (the jobs we manage), *ignored* (the OS and
   everyday apps, which shrink when someone needs room), and *others*.
3. **Ask before loading.** `room_for(needs_gb)` says whether a model fits
   beside the others, and if not, who is in the way.
4. **Wait, never fight.** A job waits and reports why. The gate never closes,
   pauses or slows anyone else's program, and scheduling is priority-aware: a
   person at the keyboard always comes before any background job.
5. **Time limit.** A wait ends after a cap (20 minutes by default). A wrong
   reading must never stall a whole queue.
6. **Don't guess.** With no fresh reading it lets the job go ahead.

## Design decisions (and what broke)

- **The card's total is the truth.** Per-program readings overlap, because
  shared memory is counted once per program. On an idle desktop, two apps
  added up to far more than the card reported, and jobs waited for room that
  was already there. Free space is now `total - min(others, used - ours) - margin`.
- **Ignore what shrinks.** The desktop compositor, driver helpers and
  everyday apps release memory when asked, so waiting for them only stalls jobs.
- **Integrated GPUs aren't "the card."** The main card is the one with the most
  memory in use, and it only changes if another card passes it by 2 GB.
- **A cap on waiting.** An early version waited forever behind a bad reading
  and stalled a queue overnight. Now it gives up after the time limit and
  goes ahead once per job.

## Example

```python
from gpugate import Gate

gate = Gate(
    total_gb=16,
    is_ours=lambda pid: pid in my_model_server_pids,
    classify=lambda pid: (program_name(pid), friendly_name(pid)),
    ignore={"dwm.exe", "explorer.exe", "chrome.exe"},
)

result = gate.wait_for_room(needs_gb=9.5, on_wait=print)   # "room", "timeout" or "cancelled"
if result != "cancelled":
    load_model()
```

## Tests

```
pip install pytest
pytest
```

Every reading in the tests is made up, so no GPU or Windows is needed.

## Status

Reading GPU memory works on Windows only (`Get-Counter`). The decision logic
is plain Python and takes any reader you give it, so adding Linux
(`nvidia-smi`, `rocm-smi`) means writing one function.

MIT licensed.
