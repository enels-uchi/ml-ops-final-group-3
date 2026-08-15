"""Lightweight console progress heartbeat for long-running steps."""
import threading
import time
from contextlib import contextmanager


@contextmanager
def heartbeat(label: str, interval: float = 15.0):
    """Prints an elapsed-time heartbeat every `interval` seconds while the wrapped
    block runs, so long CPU-bound calls with no native progress output don't look
    like they've hung.
    """
    start = time.time()
    stop_event = threading.Event()

    def _tick():
        while not stop_event.wait(interval):
            elapsed = time.time() - start
            print(f"[{label}] still running... {elapsed:.0f}s elapsed", flush=True)

    thread = threading.Thread(target=_tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        print(f"[{label}] done in {time.time() - start:.1f}s", flush=True)
