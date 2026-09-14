"""Construct the pre-executed bake-off notebook."""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

nb = new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

cells = []

def md(text: str) -> None:
    cells.append(new_markdown_cell(text))

def code(text: str) -> None:
    cells.append(new_code_cell(text.strip() + "\n"))

md(
    """# Goal-directed molecule generation bake-off

**Classical SELFIES-GRU vs hybrid 6-qubit QCBM** on public drug seeds.

This notebook is the full protocol: same seeds, same attempt budget, same Morgan Tanimoto filter, same QED objective. It is a *portfolio experiment on a simulator*, not a claim that quantum computers design drugs.

**What to look for**

1. Both generators run end-to-end and emit valid molecules.
2. Kept analogs (Tanimoto ≥ 0.40 to the seed) are ranked by ΔQED.
3. The quantum piece is a real PennyLane circuit (6-qubit `Rot` + ring `CNOT` ansatz on `default.qubit`) whose Born probabilities steer analog decoding.
"""
)

md(
    """## 0. Setup

Imports, reproducible RNG, and paths that work whether the kernel cwd is the repo root or `notebooks/`.
"""
)

code(
    """
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from IPython.display import display

HERE = Path.cwd().resolve()
ROOT = HERE if (HERE / "src").is_dir() else HERE.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    GRU_EPOCHS,
    N_ATTEMPTS,
    N_QUBITS,
    QCBM_LAYERS,
    QCBM_STEPS,
    RANDOM_SEED,
    SIMILARITY_THRESHOLD,
)
from src.data import load_reference, load_seeds
from src.hybrid_quantum import make_qcbm, random_params
from src.metrics import top_kept
from src.protocol import neighborhood_preview, run_bakeoff
from src import viz

viz.setup_style()
pd.set_option("display.precision", 3)
pd.set_option("display.max_colwidth", 80)

print("repo", ROOT)
print("attempts / method / seed:", N_ATTEMPTS)
print("Tanimoto keep threshold:", SIMILARITY_THRESHOLD)
print("GRU epochs:", GRU_EPOCHS, "| QCBM steps:", QCBM_STEPS, "| qubits:", N_QUBITS)
print("PennyLane", qml.__version__)
"""
)

md(
    """## 1. Seeds and public reference corpus

Six widely documented small-molecule drugs, taken from the public [`HIPS/molecule-autoencoder`](https://github.com/HIPS/molecule-autoencoder) `all_drugs.smi` list (see `data/README.md`). They span **low → high QED** so the task is not “everything is already maxed out.”

The reference set is the same list after drug-like filters (~1k organic, single-fragment structures). It is the GRU training corpus **and** the novelty blacklist.
"""
)

code(
    """
seeds = load_seeds()
library = load_reference()
seed_df = pd.DataFrame(
    {
        "name": [s.name for s in seeds],
        "smiles": [s.smiles for s in seeds],
        "QED": [s.qed for s in seeds],
        "SA": [s.sa for s in seeds],
        "heavy_atoms": [s.mol.GetNumHeavyAtoms() for s in seeds],
    }
)
display(seed_df)
print(f"reference molecules: {len(library)}")
display(neighborhood_preview(seeds, library))
path = viz.save_seed_grid(seeds)
display(Image.open(path))
"""
)

md(
    """## 2. Fair protocol

For **each seed**, **each method** does the following with a locked random seed:

| Step | Rule |
| --- | --- |
| Propose | exactly `N_ATTEMPTS` strings (160) |
| Valid | RDKit parse + sanitize |
| Unique | canonical SMILES |
| Novel | not in the public reference set |
| Similar | Morgan r=2 Tanimoto ≥ 0.40 vs seed, and not identical to the seed |
| Score | QED; report ΔQED and Ertl SA |

**Classical path:** pretrain a SELFIES GRU on the corpus, fine-tune on the seed neighborhood (+ a small enumerated analog cloud), sample with seed-prefix completion.

**Hybrid path:** fit a 6-bit PCA fingerprint encoder on the corpus, train a 6-qubit QCBM on the QED-weighted neighborhood distribution, sample bitstrings, decode with bitstring-conditioned analog mutations of the seed / nearest neighbors.

The protocol equalizes **budget and filters**. It does not pretend the two inductive biases are the same: the GRU is a chemical language model; the QCBM is a local latent sampler on a simulator.
"""
)

md(
    r"""## 3. The quantum circuit (6 qubits)

`Rot` on each wire plus a ring of `CNOT`s, repeated for 4 layers (the usual strongly-entangling pattern). The Born machine output is a probability table over \(2^6 = 64\) bitstrings. That is small enough for an exact statevector simulator and large enough to be a real variational circuit, not a coin flip in disguise.
"""
)

code(
    """
circuit = make_qcbm(n_qubits=N_QUBITS, n_layers=QCBM_LAYERS)
params = random_params(n_qubits=N_QUBITS, n_layers=QCBM_LAYERS, seed=RANDOM_SEED)
print(qml.draw(circuit, max_length=120)(params)[:1800])
print("parameter tensor", np.shape(params), "n_params", int(np.size(params)))

fig, ax = qml.draw_mpl(circuit, decimals=None)(params)
ax.set_title("QCBM ansatz — Rot + ring CNOTs (6 wires, 4 layers)")
viz.savefig(fig, "qcbm_circuit.png")
display(fig)
plt.close(fig)
"""
)

md(
    """## 4. Run the bake-off

This cell trains the GRU, fits every per-seed QCBM, generates both candidate pools, and writes `figures/metrics_*.csv`. On a CPU VM it should finish in a few minutes.
"""
)

code(
    """
bundle = run_bakeoff(
    n_attempts=N_ATTEMPTS,
    sim_threshold=SIMILARITY_THRESHOLD,
    seed=RANDOM_SEED,
)
metrics = bundle["metrics"]
agg = bundle["aggregate"]
results = bundle["results"]
print("GRU pretrain NLL by epoch:", [round(x, 3) for x in bundle["trainer"].losses])
display(metrics)
display(agg)
"""
)

md(
    """## 5. Training curves

Left: chemical language model. Right: quantum latent model. Both should descend; neither is trained to exhaustion on purpose — the budget class is “notebook on a laptop.”
"""
)

code(
    """
fig = viz.plot_gru_loss(bundle["trainer"].losses, extra=None)
viz.savefig(fig, "gru_pretrain_loss.png")
display(fig)
plt.close(fig)

fig = viz.plot_qcbm_losses(bundle["qcbm_losses"])
viz.savefig(fig, "qcbm_kl_loss.png")
display(fig)
plt.close(fig)
"""
)

md(
    """## 6. Head-to-head metrics

Validity / uniqueness / novelty are computed on the raw attempt budget. ΔQED and Tanimoto are computed on the **kept** set (valid, unique, similar, not the seed). Empty keep-sets show up as NaN — that is a real outcome, not a plotting bug.
"""
)

code(
    """
fig = viz.plot_metric_bars(metrics)
viz.savefig(fig, "quality_bars.png")
display(fig)
plt.close(fig)

fig = viz.plot_delta_qed(metrics)
viz.savefig(fig, "delta_qed_by_seed.png")
display(fig)
plt.close(fig)

fig = viz.plot_scatter(results)
viz.savefig(fig, "similarity_vs_delta_qed.png")
display(fig)
plt.close(fig)
"""
)

md(
    """## 7. Molecule grids — seed vs top kept analogs

Each panel is stacked: **seed** (top), **top-3 classical by ΔQED**, **top-3 hybrid by ΔQED**. If a method has no analog with Tanimoto ≥ 0.40, the row says so instead of inventing a structure.
"""
)

code(
    """
from collections import defaultdict

by_seed = defaultdict(dict)
for res in results:
    by_seed[res.seed_name][res.method] = res

for seed in bundle["seeds"]:
    classical = by_seed[seed.name]["classical_gru"]
    hybrid = by_seed[seed.name]["hybrid_qcbm"]
    img = viz.molecule_grid(seed, classical, hybrid, k=3)
    out = ROOT / "figures" / f"grid_{seed.name}.png"
    img.save(out)
    print(seed.name, "classical kept", len(classical.kept_cands), "hybrid kept", len(hybrid.kept_cands))
    display(img)
"""
)

md(
    """## 8. Top improved analog per method

A compact table for the README / portfolio caption. SMILES are canonical RDKit strings.
"""
)

code(
    """
rows = []
for seed in bundle["seeds"]:
    for method in ("classical_gru", "hybrid_qcbm"):
        tops = top_kept(by_seed[seed.name][method], k=1)
        if not tops:
            rows.append({"seed": seed.name, "method": method, "smiles": None, "delta_qed": np.nan, "tanimoto": np.nan, "qed": np.nan, "sa": np.nan})
            continue
        c = tops[0]
        rows.append(
            {
                "seed": seed.name,
                "method": method,
                "smiles": c.smiles,
                "delta_qed": c.delta_qed,
                "tanimoto": c.tanimoto,
                "qed": c.qed,
                "sa": c.sa,
            }
        )
top_df = pd.DataFrame(rows)
top_df.to_csv(ROOT / "figures" / "top_analog_per_method.csv", index=False)
display(top_df)
"""
)

md(
    """## 9. How to read the result (and how not to)

**Legitimate conclusions**

- A 6-qubit Born machine can be wired into a *goal-directed analog loop* with the same filters as a neural language model.
- On this public, small-scale protocol you can compare validity, locality (Tanimoto), and ΔQED without moving the goalposts.
- SELFIES makes the classical generator chemically valid, so the bake-off is about **neighborhood control and property gain**, not SMILES parentheses.

**Not legitimate conclusions**

- Quantum advantage, drug discovery, or “QCBM beat Big Pharma.”
- That QED-improved analogs would be better medicines (they often would not).
- That 6 simulator qubits extend to 50-qubit chemistry on hardware.

The hybrid decoder is *local by construction*; the GRU is local only insofar as prefix locking and fine-tuning work. If the GRU keep-set is smaller, that is a bias difference the protocol is designed to expose, not hide.
"""
)

md(
    """## 10. Re-run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/goal_directed_mol_gen_bakeoff.ipynb
```

Pinned stack: RDKit, PyTorch (CPU is enough), PennyLane `default.qubit`, SELFIES, scikit-learn PCA, matplotlib. No proprietary data, no QPU credentials.
"""
)

nb.cells = cells
out = Path("/workspace/notebooks/goal_directed_mol_gen_bakeoff.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("wrote", out, "cells", len(cells))
