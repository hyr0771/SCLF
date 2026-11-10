import argparse
import os
from pathlib import Path
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from sclf import SCLF
from sclf.metrics import score

ROOT = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load_data(lr, device):
    if Path(lr).name != lr:
        raise ValueError("Invalid LR identifier")
    with np.load(ROOT / "data/visium/shared.npz", allow_pickle=False) as data:
        features, paths = data["features"].copy(), data["paths"].copy()
        batch = {
            name: torch.as_tensor(data[name].copy(), dtype=torch.long, device=device)
            for name in ("regions", "knn_idx")
        }
    with np.load(
        ROOT / "data/visium/labels" / (lr + ".npz"), allow_pickle=False
    ) as data:
        splits = {
            name: (
                torch.as_tensor(
                    data[name + "_pairs"].copy(), dtype=torch.long, device=device
                ),
                torch.as_tensor(
                    data[name + "_labels"].copy(), dtype=torch.float32, device=device
                ),
            )
            for name in ("train", "valid", "test")
        }
    return features, paths, batch, splits


def train(
    lr_pair,
    seed,
    epochs=100,
    learning_rate=1e-4,
    optimizer_class=torch.optim.AdamW,
    decay=1e-4,
):
    # epochs = 200  # Hippocampus、PDAC
    # epochs = 150  # Olfactory bulb
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, paths, batch, splits = load_data(lr_pair, device)
    labels = {name: values[1].cpu().numpy() for name, values in splits.items()}
    if not len(labels["valid"]):
        raise ValueError("A nonempty validation split is required")
    model = SCLF(features, paths, int(batch["regions"].max()) + 1).to(device)
    optimizer = optimizer_class(
        model.parameters(), lr=learning_rate, weight_decay=decay
    )
    output = ROOT / "outputs" / lr_pair / ("seed" + str(seed))
    output.mkdir(parents=True, exist_ok=True)
    best_value, best_state = -float("inf"), None
    pairs, targets = splits["train"]
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(pairs), device=device)
        for offset in range(0, len(pairs), 4096):
            chosen = order[offset : offset + 4096]
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch, pairs[chosen], targets[chosen])["total_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        probabilities = model.score_splits(batch, {"valid": splits["valid"]})["valid"]
        valid = score(labels["valid"], probabilities)
        value = valid["auc"]
        if value is None:
            p = np.clip(probabilities, 1e-7, 1 - 1e-7)
            y = labels["valid"]
            value = float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if value > best_value + 1e-6:
            best_value, best_state = value, model.compact_state()
            torch.save(best_state, output / "best_model.pt")
        print(
            f"epoch {epoch + 1}/{epochs} valid: AUC={valid['auc']} "
            f"ACC={valid['acc']:.3f} F1={valid['f1']:.3f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_weights(best_state)
    probabilities = model.score_splits(
        batch, {"valid": splits["valid"], "test": splits["test"]}
    )
    for name, values in probabilities.items():
        metrics = score(labels[name], values)
        print(
            f"{name}: AUC={metrics['auc']} "
            f"ACC={metrics['acc']:.3f} F1={metrics['f1']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SCLF on the prepared Visium data"
    )
    parser.add_argument("--lr-pair", default="Mif_Epha5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer", choices=("AdamW", "Adam", "SGD"), default="AdamW"
    )
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--epochs", "--epoch", type=int, default=100)
    args = parser.parse_args()
    train(
        args.lr_pair,
        args.seed,
        epochs=args.epochs,
        learning_rate=args.lr,
        optimizer_class=getattr(torch.optim, args.optimizer),
        decay=args.decay,
    )
