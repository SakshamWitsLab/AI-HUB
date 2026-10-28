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
    from backend import runner

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
