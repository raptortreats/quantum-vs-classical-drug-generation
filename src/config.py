"""Shared protocol constants for the goal-directed bake-off."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
NOTEBOOK_DIR = ROOT / "notebooks"

RANDOM_SEED = 42

# Fair comparison budget
N_ATTEMPTS = 160
SIMILARITY_THRESHOLD = 0.40
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# Classical GRU
GRU_EMBED_DIM = 48
GRU_HIDDEN_DIM = 96
GRU_LAYERS = 2
GRU_DROPOUT = 0.2
GRU_EPOCHS = 18
GRU_BATCH_SIZE = 64
GRU_LR = 2e-3
GRU_MAX_LEN = 72
GRU_RANDOM_SMILES = 4
GRU_FINETUNE_EPOCHS = 6
GRU_TEMPERATURE = 0.85

# Hybrid QCBM
N_QUBITS = 6
QCBM_LAYERS = 4
QCBM_STEPS = 70
QCBM_LR = 0.12
QCBM_NEIGHBOR_K = 40
QCBM_QED_TEMP = 4.0
