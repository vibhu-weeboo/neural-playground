# Neural Playground 🧠✨

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#getting-started) [![Framework](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/) [![CI](https://img.shields.io/badge/Status-Experimental-purple.svg)](#about)

An interactive sandbox for tiny neural network experiments. Spin up small demos, tweak hyperparameters, and visualize what networks learn—in seconds.

## What’s inside
- demo.py: A minimal PyTorch XOR demo with live decision boundary + loss plots
- requirements.txt: Dependencies for quick setup
- README: You’re reading it!

## Cool Demo: XOR Decision Boundary 🔴🔵
Train a tiny MLP to learn the classic XOR dataset and watch the decision boundary evolve during training.

Placeholder screenshots:
- ![Decision boundary placeholder](docs/img/decision_boundary.png)
- ![Loss curve placeholder](docs/img/loss_curve.png)

(You can replace these with real screenshots after running the demo.)

## Getting Started

### 1) Clone the repo
```
git clone https://github.com/vibhu-weeboo/neural-playground.git
cd neural-playground
```

### 2) Create a virtual environment (recommended)
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3) Install dependencies
```
pip install -r requirements.txt
```

If you don’t have a GPU, PyTorch will install the CPU build automatically via pip.

### 4) Run the XOR demo
```
python demo.py --epochs 1000 --hidden 8 --lr 0.1 --seed 42
```
Optional flags:
- --batch-size: default 4
- --hidden: hidden layer size (default 8)
- --lr: learning rate (default 0.1)
- --epochs: number of epochs (default 1000)
- --seed: random seed (default 42)

When running, two live matplotlib plots appear:
- Decision boundary over the input space
- Training loss across epochs

## Example Output
```
Final accuracy: 100.0%
```
Accuracy may vary slightly depending on seed and hyperparameters.

## Troubleshooting
- If plots don’t appear, ensure you’re not in a headless environment and have a GUI backend available for matplotlib.
- On WSL, you may need an X server or run in VS Code with the Python extension.

## Contributing
Ideas for new micro-demos welcome! Examples:
- Spiral classification
- Simple autoencoder
- Tiny CNN on MNIST subset

## License
MIT
