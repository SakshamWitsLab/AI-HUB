"""One-shot CLI to run a single OCR job in an isolated subprocess.

Usage: python -m backend.runner_cli <config_json_path>
Prints a JSON result object on stdout. If the process crashes (e.g. Metal
errors), the hub survives and reports the failure cleanly.

GPU note: when device is MPS, an flock serializes runs across processes;
concurrent {{Metal encoder}} access crashes, so models queue politely.
"""
import contextlib
import json
import sys

CONFIG = json.loads(open(sys.argv[1]).read())

# Cap BLAS/OMP thread pools in the runner subprocess — without this, each of
# the hub's parallel runners spawns one thread per core and the container's
# CPU/memory usage explodes under multi-page runs.
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OCR_HUB_MAX_THREADS", "4"))


@contextlib.contextmanager
def maybe_mps_lock():
    if CONFIG.get("device") != "mps":
        yield
        return
    import fcntl

    with open("/tmp/ocr-hub-mps.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def main():
    import torch

    from backend import runner

    # respect OCR_HUB_MAX_THREADS on top of the env vars (torch reads OMP too,
    # but intra-op pool would default to core count)
    try:
        torch.set_num_threads(int(os.environ.get("OCR_HUB_MAX_THREADS", "4")))
    except Exception:
        pass

    blocks = []
    with maybe_mps_lock():
        blocks = runner.dispatch(
            CONFIG["engine"],
            CONFIG["image_path"],
            device=CONFIG["device"],
            config={k: v for k, v in CONFIG.items() if k not in ("engine", "device", "image_path")},
        )
    print(json.dumps({"blocks": blocks, "crash": None}, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surface as JSON, let parent live
        print(json.dumps({"crash": f"{type(exc).__name__}: {str(exc)[:400]}"}))
        sys.exit(3)
