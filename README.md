# Neural Net From Scratch

A feed-forward neural network implemented from scratch (pure Python + NumPy, manual
backpropagation) and trained on MNIST — plus experimental variants, including an
evolution-based optimizer and a no-backprop baseline, to compare training methods.

## Versions
| File | What it explores |
|---|---|
| `AIMS1.0.0 (with backpropagation).py` | The main backprop implementation |
| `AIMS1.0.0(no back propagation).py` | Baseline without backprop |
| `AIMS1.1.0 evolution.py` | Evolutionary weight optimization |
| `AIMS1.0.0 Erode.py`, `MSEtest.py` | Experiments |

## Data
The MNIST dataset is **not committed** — download it and point the scripts at it.

## Run it
```bash
python -m pip install numpy colorama
python "AIMS1.0.0 (with backpropagation).py"
```
