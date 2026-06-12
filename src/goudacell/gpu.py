"""GPU detection and diagnostics.

A bare ``torch.cuda.is_available()`` returns ``False`` for several unrelated
reasons (no GPU on the node, a CPU-only torch wheel, a driver/runtime
mismatch). On the cluster the usual cause is simply running on the head node or
forgetting ``--gres=gpu:1``. This module turns that one bare boolean into an
actionable diagnosis so the notebook can tell the user *why* the GPU is missing.
"""

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUStatus:
    """Result of a GPU availability check.

    Attributes:
        available: Whether torch can actually use a CUDA GPU.
        device_name: Name of the active CUDA device, or None.
        torch_cuda_build: CUDA version torch was built against (e.g. "12.6"),
            or None for a CPU-only wheel / when torch is missing.
        driver_detected: Whether an NVIDIA driver is visible (``nvidia-smi``
            runs and reports at least one GPU).
        reason: Human-readable explanation of the result, with the fix when the
            GPU is unavailable.
    """

    available: bool
    device_name: Optional[str]
    torch_cuda_build: Optional[str]
    driver_detected: bool
    reason: str

    def __str__(self) -> str:
        """Return a one-line summary suitable for printing in a notebook."""
        mark = "✓" if self.available else "✗"
        return f"{mark} GPU: {self.reason}"


def _nvidia_smi_detects_gpu() -> bool:
    """Return True if ``nvidia-smi`` is present and lists at least one GPU."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def detect_gpu() -> GPUStatus:
    """Check whether a usable CUDA GPU is available and explain the result.

    Returns:
        A :class:`GPUStatus` describing availability and, when the GPU is not
        usable, the most likely cause and how to fix it.
    """
    try:
        import torch
    except ImportError:
        return GPUStatus(
            available=False,
            device_name=None,
            torch_cuda_build=None,
            driver_detected=_nvidia_smi_detects_gpu(),
            reason=(
                "PyTorch is not installed. Install a GPU build with "
                "`uv pip install -e '.[cellpose4]'`."
            ),
        )

    cuda_build = torch.version.cuda
    driver_detected = _nvidia_smi_detects_gpu()

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return GPUStatus(
            available=True,
            device_name=name,
            torch_cuda_build=cuda_build,
            driver_detected=True,
            reason=f"{name} (torch CUDA {cuda_build})",
        )

    # CUDA is not available — figure out why.
    if not driver_detected:
        reason = (
            "No NVIDIA GPU visible on this node. Run on a GPU node via SLURM, "
            "e.g. `sbatch scripts/jupyter_gpu.sh` (which requests --gres=gpu:1), "
            "and make sure the notebook kernel is running there."
        )
    elif cuda_build is None:
        reason = (
            "A GPU is present but torch is a CPU-only build (torch.version.cuda "
            "is None). Reinstall a CUDA build: `uv pip install -e '.[cellpose4]'`."
        )
    else:
        reason = (
            f"A GPU is present and torch was built for CUDA {cuda_build}, but "
            "torch.cuda.is_available() is False — likely a driver/runtime "
            "mismatch. Check `nvidia-smi` driver vs. torch CUDA version."
        )

    return GPUStatus(
        available=False,
        device_name=None,
        torch_cuda_build=cuda_build,
        driver_detected=driver_detected,
        reason=reason,
    )


def resolve_gpu(requested: bool = True, *, verbose: bool = True) -> bool:
    """Resolve whether to use the GPU, falling back to CPU when unavailable.

    Use this in place of a hardcoded ``GPU = True`` so a missing GPU degrades to
    a CPU run with a clear warning instead of a confusing failure.

    Args:
        requested: Whether GPU use is desired.
        verbose: Whether to print the diagnosis when falling back to CPU.

    Returns:
        True if the GPU should and can be used, otherwise False.
    """
    if not requested:
        return False

    status = detect_gpu()
    if not status.available and verbose:
        print(f"⚠️  GPU requested but unavailable — falling back to CPU.\n   {status.reason}")
    return status.available
