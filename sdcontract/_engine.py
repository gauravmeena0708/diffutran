"""diffutran engine — ported from the host's wrappers/diffutran_wrapper.py.

Standalone (no host `wrappers/` imports) so the plugin is self-contained inside
the submodule. Training is unconditional (num_classes=None); constraints are
applied at inference via REPAINT-style imputation, so train/generate decouple:
train once -> model_dir, then generate under N constraints from the same dir.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# diffutran source lives in the submodule (sub/diffutran/src).
DIFFUTRAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIFFUTRAN_ROOT / "src"))

from common.data_utils import DataPreprocessor  # noqa: E402
from diffutron.model import DiffusionModel  # noqa: E402
from diffutron.diffusion import GaussianDiffusion  # noqa: E402

logger = logging.getLogger("diffutran.sdcontract")

CKPT_NAME = "checkpoint_epoch_final.pt"
PREPROC_NAME = "preprocessor.json"
META_NAME = "meta.json"
TRAIN_LOCK_DIR = ".train_lock"
NUM_TIMESTEPS = 1000
D_MODEL = 256
NUM_LAYERS = 6
NHEAD = 8


def resolve_device(device: str | None) -> str:
    if device and device.startswith("cuda") and torch.cuda.is_available():
        return device if ":" in device else "cuda"
    return "cuda" if (not device or device.startswith("cuda")) and torch.cuda.is_available() else "cpu"


class _ModelDirLock:
    def __init__(self, model_dir: Path, poll_seconds: float = 1.0, timeout_seconds: float = 1800.0):
        self.lock_dir = model_dir / TRAIN_LOCK_DIR
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def __enter__(self):
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                os.mkdir(self.lock_dir)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Could not acquire training lock {self.lock_dir}")
                time.sleep(self.poll_seconds)

    def __exit__(self, exc_type, exc, tb):
        try:
            os.rmdir(self.lock_dir)
        except FileNotFoundError:
            pass


class _SimpleDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _infer_categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _build_model(num_numerical, cat_cardinalities, device):
    return DiffusionModel(
        num_numerical_features=num_numerical,
        categorical_cardinalities=cat_cardinalities,
        d_model=D_MODEL, num_layers=NUM_LAYERS, nhead=NHEAD,
        num_classes=None,  # unconditional — every column lives in the feature tensor
    ).to(device)


def train(req: dict) -> Path:
    """Train iff the checkpoint is missing; always leaves a contract meta.json.

    Returns the model_dir. Writes ``meta.json`` carrying the required contract
    fields plus diffutran's model config (read back by ``generate``).
    """
    model_dir = Path(req["output_model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt = model_dir / CKPT_NAME
    device = resolve_device(req.get("device"))
    num_epochs = int(req.get("num_epochs", 50))

    with _ModelDirLock(model_dir):
        if ckpt.exists():
            logger.info("Found checkpoint at %s — skipping training", ckpt)
            return model_dir

        df = pd.read_csv(req["train_csv"])
        cat_cols = _infer_categorical_cols(df)
        preprocessor = DataPreprocessor()
        data = preprocessor.fit_transform(df, categorical_cols=cat_cols)
        preprocessor.save(str(model_dir / PREPROC_NAME))

        meta = {
            "schema_version": "1.0",
            "method": req["method"],
            "dataset": req["dataset"],
            "label_column": req.get("label_column"),
            "columns": list(df.columns),
            "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
            "sdcontract_core_version": "1.0",
            # diffutran model config (consumed by generate / _load):
            "cat_cols": cat_cols,
            "total_dim": int(data.shape[1]),
            "num_numerical": len(preprocessor.numerical_cols),
            "categorical_cardinalities": preprocessor.categorical_cardinalities,
            "num_epochs": num_epochs,
        }
        (model_dir / META_NAME).write_text(json.dumps(meta, indent=2))

        loader = DataLoader(_SimpleDataset(data), batch_size=int(req.get("train_batch_size", 128)),
                            shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
        model = _build_model(meta["num_numerical"], meta["categorical_cardinalities"], device)
        diffusion = GaussianDiffusion(num_timesteps=NUM_TIMESTEPS).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader) * num_epochs)

        logger.info("Training %d epochs on %d rows (%d dims) device=%s",
                    num_epochs, len(df), data.shape[1], device)
        model.train()
        for epoch in range(num_epochs):
            for batch_x in loader:
                batch_x = batch_x.to(device)
                t = torch.randint(0, NUM_TIMESTEPS, (batch_x.shape[0],), device=device).long()
                loss = diffusion.p_losses(model, batch_x, t, condition=None)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        torch.save({"model_state_dict": model.state_dict(), "meta": meta}, str(model_dir / CKPT_NAME))
        logger.info("Checkpoint saved -> %s", model_dir / CKPT_NAME)
    return model_dir


def _load(model_dir: Path, device: str):
    meta = json.loads((model_dir / META_NAME).read_text())
    preprocessor = DataPreprocessor()
    preprocessor.load(str(model_dir / PREPROC_NAME))
    model = _build_model(meta["num_numerical"], meta["categorical_cardinalities"], device)
    ckpt = torch.load(str(model_dir / CKPT_NAME), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = GaussianDiffusion(num_timesteps=NUM_TIMESTEPS).to(device)
    return model, diffusion, preprocessor, meta


def _split_constraints(native_constraints, preprocessor):
    """Native ``col=val`` strings -> (categorical dict, numerical dict)."""
    cat_c, num_c = {}, {}
    for raw in native_constraints:
        if "=" not in raw:
            continue
        col, val = (s.strip() for s in raw.split("=", 1))
        if col in preprocessor.categorical_cols:
            cat_c[col] = val
        elif col in preprocessor.numerical_cols:
            num_c[col] = float(val)
        else:
            logger.warning("Constraint column %r not in preprocessor — skipped", col)
    return cat_c, num_c


def generate(req: dict, native_constraints: list[str]) -> Path:
    """Sample to the host-chosen output_csv_path under the given native (col=val)
    constraints; never retrains."""
    device = resolve_device(req.get("device"))
    model_dir = Path(req["model_dir"])
    n_samples = int(req["n_samples"])
    gen_batch_size = int(req.get("gen_batch_size", 512))
    num_inference_steps = int(req.get("num_inference_steps", 50))
    eta = float(req.get("eta", 0.0))

    model, diffusion, preprocessor, meta = _load(model_dir, device)
    total_dim = meta["total_dim"]
    cat_c, num_c = _split_constraints(native_constraints, preprocessor)

    if cat_c or num_c:
        logger.info("Categorical constraints: %s | Numerical: %s", cat_c, num_c)
        cx0_full, cmask_full = preprocessor.get_constraint_tensors(
            {**cat_c, **num_c}, batch_size=gen_batch_size, device=device)
    else:
        logger.info("No constraints — generating unconditionally")
        cx0_full, cmask_full = None, None

    tensors, remaining = [], n_samples
    while remaining > 0:
        bs = min(gen_batch_size, remaining)
        cx0 = cx0_full[:bs] if cx0_full is not None else None
        cmask = cmask_full[:bs] if cmask_full is not None else None
        x = diffusion.sample(model, shape=(bs, total_dim), condition=None, guidance_scale=1.0,
                             constraint_x0=cx0, constraint_mask=cmask, device=device,
                             num_inference_steps=num_inference_steps, eta=eta)
        tensors.append(x.cpu())
        remaining -= bs

    df_out = preprocessor.inverse_transform(torch.cat(tensors, dim=0)[:n_samples])
    for col, val in cat_c.items():  # pin categoricals for exact satisfaction
        df_out[col] = val

    out_path = Path(req["output_csv_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info("Saved %d samples -> %s", len(df_out), out_path)
    return out_path
