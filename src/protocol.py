"""End-to-end fair bake-off: classical GRU vs hybrid QCBM."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from .chemistry import canonical_smiles, enumerate_analogs, qed_score, sa_score
from .classical import finetune_gru, generate_smiles, train_gru
from .config import (
    FIGURE_DIR,
    GRU_EPOCHS,
    GRU_FINETUNE_EPOCHS,
    GRU_RANDOM_SMILES,
    N_ATTEMPTS,
    QCBM_NEIGHBOR_K,
    QCBM_STEPS,
    RANDOM_SEED,
    SIMILARITY_THRESHOLD,
)
from .data import MoleculeRecord, load_reference, load_seeds, reference_smiles_set
from .hybrid_quantum import fit_bit_encoder, generate_hybrid, neighborhood
from .metrics import MethodResult, aggregate_table, evaluate_generations, results_table


def _clone_trainer(trainer):
    """Shallow-copy model weights so per-seed fine-tunes do not leak."""
    import torch

    cloned = copy.copy(trainer)
    cloned.model = copy.deepcopy(trainer.model)
    cloned.losses = list(trainer.losses)
    cloned.device = trainer.device
    cloned.vocab = trainer.vocab
    # silence unused
    _ = torch
    return cloned


def run_bakeoff(
    n_attempts: int = N_ATTEMPTS,
    sim_threshold: float = SIMILARITY_THRESHOLD,
    seed: int = RANDOM_SEED,
    gru_epochs: int | None = None,
    qcbm_steps: int | None = None,
) -> dict[str, Any]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    seeds = load_seeds()
    library = load_reference()
    ref_set = reference_smiles_set(library)

    n_gru_epochs = GRU_EPOCHS if gru_epochs is None else gru_epochs
    q_steps = QCBM_STEPS if qcbm_steps is None else qcbm_steps

    trainer = train_gru(
        library,
        n_random=GRU_RANDOM_SMILES,
        epochs=n_gru_epochs,
        seed=seed,
    )
    encoder = fit_bit_encoder(library, seed=seed)

    results: list[MethodResult] = []
    ft_losses: dict[str, list[float]] = {}
    qcbm_losses: dict[str, list[float]] = {}

    for seed_mol in seeds:
        neigh_recs = [r for r, _ in neighborhood(seed_mol, library, k=QCBM_NEIGHBOR_K)]
        analog_recs = []
        for mol in enumerate_analogs(seed_mol.mol, limit=16):
            smi = canonical_smiles(mol)
            if not smi:
                continue
            analog_recs.append(
                MoleculeRecord(
                    name=f"{seed_mol.name}_analog",
                    smiles=smi,
                    source="seed_analog_enum",
                    qed=qed_score(mol),
                    sa=sa_score(mol),
                    mol=mol,
                )
            )
        ft_pool = neigh_recs + analog_recs or [seed_mol]
        local_trainer = _clone_trainer(trainer)
        seed_offset = int(np.frombuffer(seed_mol.name.encode(), dtype=np.uint8).sum())
        ft_losses[seed_mol.name] = finetune_gru(
            local_trainer,
            ft_pool,
            epochs=GRU_FINETUNE_EPOCHS,
            seed=seed + seed_offset,
        )

        local_rng = np.random.default_rng(rng.integers(1, 1_000_000))
        classical_raw = generate_smiles(
            local_trainer,
            n=n_attempts,
            rng=local_rng,
            seed_smiles=seed_mol.smiles,
        )
        classical_res = evaluate_generations(
            classical_raw, seed_mol, "classical_gru", ref_set, sim_threshold
        )
        classical_res.extra["finetune_loss"] = ft_losses[seed_mol.name]
        results.append(classical_res)

        h_rng = np.random.default_rng(rng.integers(1, 1_000_000))
        hybrid_raw, losses, params = generate_hybrid(
            seed_mol, library, encoder, n=n_attempts, rng=h_rng, steps=q_steps
        )
        qcbm_losses[seed_mol.name] = losses
        hybrid_res = evaluate_generations(
            hybrid_raw, seed_mol, "hybrid_qcbm", ref_set, sim_threshold
        )
        hybrid_res.extra["qcbm_loss"] = losses
        hybrid_res.extra["n_params"] = int(np.size(params))
        results.append(hybrid_res)

    df = results_table(results)
    agg = aggregate_table(df)
    df.to_csv(FIGURE_DIR / "metrics_by_seed.csv", index=False)
    agg.to_csv(FIGURE_DIR / "metrics_aggregate.csv", index=False)
    return {
        "seeds": seeds,
        "library": library,
        "trainer": trainer,
        "encoder": encoder,
        "results": results,
        "metrics": df,
        "aggregate": agg,
        "qcbm_losses": qcbm_losses,
        "ft_losses": ft_losses,
        "n_attempts": n_attempts,
        "sim_threshold": sim_threshold,
    }


def neighborhood_preview(seeds: list[MoleculeRecord], library: list[MoleculeRecord]) -> pd.DataFrame:
    rows = []
    for s in seeds:
        neigh = neighborhood(s, library, k=QCBM_NEIGHBOR_K)
        sims = [sim for _, sim in neigh]
        rows.append(
            {
                "seed": s.name,
                "qed": s.qed,
                "sa": s.sa,
                "top_k": len(neigh),
                "max_sim": max(sims) if sims else np.nan,
                "mean_sim": float(np.mean(sims)) if sims else np.nan,
            }
        )
    return pd.DataFrame(rows)
