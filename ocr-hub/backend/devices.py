"""Detect available compute devices: CUDA (NVIDIA), MPS (Apple Silicon), CPU."""
import shutil
import subprocess
import sys

try:
    import torch
except ImportError:
    torch = None


def detect() -> dict:
    devices = {
        "cpu": {"available": True, "label": "CPU", "detail": "Always available"},
        "cuda": {"available": False, "label": "CUDA (NVIDIA)", "detail": "Not detected"},
        "mps": {"available": False, "label": "MPS (Apple GPU)", "detail": "Not detected"},
    }

    if torch is not None:
        try:
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                devices["cuda"] = {"available": True, "label": f"CUDA (NVIDIA)", "detail": name}
        except Exception:
            pass
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                devices["mps"] = {"available": True, "label": "MPS (Apple GPU)", "detail": "Metal Performance Shaders"}
        except Exception:
            pass
    else:
        # Heuristics without torch installed
        if sys.platform == "darwin":
            try:
                r = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=15
                )
                if "Metal" in r.stdout:
                    devices["mps"] = {
                        "available": False,
                        "label": "MPS (Apple GPU)",
                        "detail": "Metal present but torch not installed — install torch in the runner",
                    }
            except Exception:
                pass
    try:
        if shutil.which("nvidia-smi"):
            devices["cuda"] = {"available": True, "label": "CUDA (NVIDIA)", "detail": "nvidia-smi present (torch may be missing)"}
    except Exception:
        pass
    return devices


def resolve(preferred: str | None = None) -> str:
    """Return a usable torch device string, honoring preference."""
    d = detect()
    if preferred and preferred != "auto" and d.get(preferred, {}).get("available"):
        return "cuda" if preferred == "cuda" else preferred
    for dev in ("cuda", "mps"):
        if d[dev]["available"]:
            return dev
    return "cpu"


if __name__ == "__main__":
    import json

    print(json.dumps(detect(), indent=2))
