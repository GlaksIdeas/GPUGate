"""Every reading here is made up: no GPU and no Windows needed."""

import pytest

from gpugate import Gate, parse_windows_counters

GB = 1024**3
CARD = "luid_0x00000000_0x0000d1c2"
IGPU = "luid_0x00000000_0x0000a001"


def proc(pid, gb, luid=CARD):
    return {"Path": f"\\\\pc\\gpu process memory(pid_{pid}_{luid}_phys_0)\\dedicated usage", "CookedValue": gb * GB}


def card(gb, luid=CARD):
    return {"Path": f"\\\\pc\\gpu adapter memory({luid}_phys_0)\\dedicated usage", "CookedValue": gb * GB}


NAMES = {200: ("music.exe", "Music Tool"), 300: ("dwm.exe", "Desktop"), 400: ("game.exe", "Some Game")}


@pytest.fixture
def gate():
    t = {"now": 1000.0}
    g = Gate(16, is_ours=lambda pid: pid == 100,
             classify=lambda pid: NAMES.get(pid, (f"p{pid}.exe", f"program {pid}")),
             ignore={"dwm.exe"}, reader=lambda: None, clock=lambda: t["now"])
    g.t = t
    return g


def test_samples_are_split_per_card_and_added_up():
    cards = parse_windows_counters([proc(100, 6), proc(100, 4), proc(300, 0.5, IGPU), card(11), card(0.5, IGPU)])
    assert cards[CARD].procs[100] == pytest.approx(10)
    assert cards[CARD].used_gb == pytest.approx(11)


def test_card_without_a_total_is_dropped():
    assert parse_windows_counters([proc(100, 2)]) == {}


def test_integrated_gpu_is_not_the_main_card(gate):
    gate.look([card(1.2), card(0.6, IGPU), proc(400, 1.0)])
    assert gate.picture().used_gb == pytest.approx(1.2)


def test_job_fits_on_an_empty_card(gate):
    gate.look([card(1.0), proc(300, 0.5)])
    assert gate.room_for(9.5) == (True, "")


def test_job_waits_for_another_program_and_names_it(gate):
    gate.look([card(11), proc(200, 9.1), proc(100, 0)])
    fits, why = gate.room_for(9.5)
    assert not fits and "Music Tool is using 9.1 GB" in why


def test_ignored_programs_never_hold_a_job_back(gate):
    gate.look([card(7), proc(300, 6.0)])
    assert gate.room_for(9.5)[0]


def test_overlapping_readings_are_capped_by_the_cards_total(gate):
    # Two programs each claim 5 GB of the same shared 5 GB; the card says 5 in use.
    gate.look([card(5), proc(200, 5), proc(400, 5)])
    assert gate.room_for(9.5)[0]


def test_stale_reading_never_holds_work_back(gate):
    gate.look([card(15), proc(200, 14)])
    gate.t["now"] += 500
    assert gate.room_for(9.5) == (True, "")


def test_wait_returns_when_room_appears(gate):
    readings = iter([[card(15), proc(200, 14)], [card(2), proc(200, 1)]])
    gate._reader = lambda: next(readings)
    messages = []
    result = gate.wait_for_room(9.5, on_wait=messages.append, sleep=lambda s: None)
    assert result == "room" and len(messages) == 1 and "Music Tool" in messages[0]


def test_wait_gives_up_after_the_time_limit(gate):
    gate._reader = lambda: [card(15), proc(200, 14)]
    messages = []

    def sleep(s):
        gate.t["now"] += s

    result = gate.wait_for_room(9.5, on_wait=messages.append, max_wait_s=60, poll_s=20, sleep=sleep)
    assert result == "timeout" and "Going ahead anyway" in messages[-1]


def test_wait_stops_on_cancel(gate):
    gate._reader = lambda: [card(15), proc(200, 14)]
    assert gate.wait_for_room(9.5, cancelled=lambda: True) == "cancelled"


def test_running_job_yields_when_others_grow_and_card_is_full(gate):
    gate.look([card(15.5), proc(100, 9), proc(200, 6)])
    assert gate.crowded(0.5) != ""
    assert gate.crowded(5.5) == ""
