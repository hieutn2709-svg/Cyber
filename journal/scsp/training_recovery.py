"""Atomic epoch-boundary recovery for trusted, locally generated training runs.

The single recovery file includes the selected best checkpoint so a crash
between two file writes cannot mix epochs during a subsequent resume.
"""
from __future__ import annotations

import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def run_lock(output_dir):
    """Reject concurrent writers; the OS releases the advisory lock on exit."""
    import fcntl

    with (Path(output_dir) / ".training.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another training writer is active in this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_save(payload, path):
    import torch

    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save(path, *, identity, model, optimizer, scaler, progress, best_checkpoint):
    import numpy as np
    import torch

    atomic_save({
        "format": "matched_epoch_recovery_v1",
        "identity": identity,
        "progress": progress,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_checkpoint": best_checkpoint,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }, path)


def load(path, identity):
    import torch

    # Only load our own trusted run artifacts; pickle is not an import format.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "matched_epoch_recovery_v1":
        raise ValueError("not a complete training recovery checkpoint")
    if state.get("identity") != identity:
        raise ValueError("training recovery identity mismatch")
    required = {"optimizer_state_dict", "scaler_state_dict", "model_state_dict", "rng", "best_checkpoint", "progress"}
    if not required.issubset(state):
        raise ValueError("incomplete training recovery checkpoint")
    progress = state["progress"]
    epoch = progress.get("epoch", 0)
    if not 1 <= epoch <= identity["epoch_budget"]:
        raise ValueError("invalid recovery epoch")
    if [row["epoch"] for row in progress["history"]] != list(range(1, epoch + 1)):
        raise ValueError("recovery history does not match epoch")
    if not 1 <= progress["best_epoch"] <= epoch or state["best_checkpoint"]["epoch"] != progress["best_epoch"]:
        raise ValueError("recovery best checkpoint does not match epoch")
    return state


def restore(state, model, optimizer, scaler, best_path):
    import numpy as np
    import torch

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    atomic_save(state["best_checkpoint"], best_path)
    # Restore RNG last, after all model construction and file operations.
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state["progress"]
