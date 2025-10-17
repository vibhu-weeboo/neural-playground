# demo.py
"""
Interactive XOR neural network demo using PyTorch.
- Builds a tiny MLP
- Trains on XOR
- Plots decision boundary live

Run:
  python demo.py --epochs 1000 --hidden 8 --lr 0.1 --seed 42

Requires:
  - torch
  - matplotlib
"""

import argparse
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Config:
    hidden: int = 8
    lr: float = 0.1
    epochs: int = 1000
    batch_size: int = 4
    seed: int = 42
    device: str = "cpu"


class TinyMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=8, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def make_xor(noise=0.0):
    # Classic XOR points
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)[:, None]
    if noise > 0:
        X = X + np.random.randn(*X.shape).astype(np.float32) * noise
    return X, y


def plot_decision_boundary(model, ax, title="Decision boundary", device="cpu"):
    model.eval()
    # Create a grid over [ -0.5, 1.5 ]
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(grid).to(device))
        probs = torch.sigmoid(logits).cpu().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, probs, levels=50, cmap="coolwarm", alpha=0.8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(title)


def train(cfg: Config):
    set_seed(cfg.seed)

    # Data
    X, y = make_xor()
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # Model
    model = TinyMLP(hidden=cfg.hidden).to(cfg.device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Live plot setup
    plt.ion()
    fig, (ax_bound, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))
    losses = []

    # Scatter true points
    ax_bound.scatter(X[y[:, 0] == 0][:, 0], X[y[:, 0] == 0][:, 1], c="k", marker="x", label="class 0")
    ax_bound.scatter(X[y[:, 0] == 1][:, 0], X[y[:, 0] == 1][:, 1], c="w", edgecolors="k", label="class 1")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(ds)
        losses.append(epoch_loss)

        # Re-plot every N steps
        if epoch % max(1, cfg.epochs // 50) == 0 or epoch == 1 or epoch == cfg.epochs:
            ax_bound.clear()
            # redraw points
            ax_bound.scatter(X[y[:, 0] == 0][:, 0], X[y[:, 0] == 0][:, 1], c="k", marker="x", label="class 0")
            ax_bound.scatter(X[y[:, 0] == 1][:, 0], X[y[:, 0] == 1][:, 1], c="w", edgecolors="k", label="class 1")
            plot_decision_boundary(model, ax_bound, title=f"Decision boundary @ epoch {epoch}")
            ax_bound.legend(loc="upper right")

            ax_loss.clear()
            ax_loss.plot(losses, label="BCE loss")
            ax_loss.set_xlabel("epoch")
            ax_loss.set_ylabel("loss")
            ax_loss.set_title("Training loss")
            ax_loss.legend()
            plt.pause(0.001)

    # Final metrics
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).numpy()
        preds = (probs > 0.5).astype(np.float32)
        acc = (preds == y).mean()
    print(f"Final accuracy: {acc*100:.1f}%")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tiny XOR neural net demo (PyTorch)")
    p.add_argument("--hidden", type=int, default=8, help="Hidden units")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    p.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    cfg = Config(hidden=args.hidden, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    train(cfg)
