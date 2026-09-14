"""Evaluation metrics and the per-seed bake-off loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .chemistry import (
    canonical_smiles,
    mol_from_smiles,
    morgan_fp,
    qed_score,
    sa_score,
    tanimoto,
)
from .config import N_ATTEMPTS, SIMILARITY_THRESHOLD
from .data import MoleculeRecord


@dataclass
class Candidate:
    smiles: str
    method: str
    seed_name: str
    valid: bool
    novel: bool
    tanimoto: Optional[float]
    qed: Optional[float]
    sa: Optional[float]
    delta_qed: Optional[float]
    kept: bool


@dataclass
class MethodResult:
    method: str
    seed_name: str
    seed_smiles: str
    seed_qed: float
    attempted: int
    candidates: list[Candidate] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def valid(self) -> list[Candidate]:
        return [c for c in self.candidates if c.valid]

    @property
    def kept_cands(self) -> list[Candidate]:
        return [c for c in self.candidates if c.kept]


def evaluate_generations(
    raw_smiles: list[str],
    seed: MoleculeRecord,
    method: str,
    reference: set[str],
    sim_threshold: float = SIMILARITY_THRESHOLD,
) -> MethodResult:
    seed_fp = morgan_fp(seed.mol)
    result = MethodResult(
        method=method,
        seed_name=seed.name,
        seed_smiles=seed.smiles,
        seed_qed=float(seed.qed),
        attempted=len(raw_smiles),
    )
    seen_valid = set()
    for smi in raw_smiles:
        mol = mol_from_smiles(smi) if smi else None
        can = canonical_smiles(mol) if mol is not None else None
        if mol is None or can is None:
            result.candidates.append(
                Candidate(
                    smiles=smi or "",
                    method=method,
                    seed_name=seed.name,
                    valid=False,
                    novel=False,
                    tanimoto=None,
                    qed=None,
                    sa=None,
                    delta_qed=None,
                    kept=False,
                )
            )
            continue
        sim = tanimoto(seed_fp, morgan_fp(mol))
        qed = qed_score(mol)
        sa = sa_score(mol)
        unique_first = can not in seen_valid
        seen_valid.add(can)
        novel = can not in reference and can != seed.smiles
        kept = (
            unique_first
            and can != seed.smiles
            and sim >= sim_threshold
            and mol.GetNumHeavyAtoms() <= seed.mol.GetNumHeavyAtoms() + 8
        )
        result.candidates.append(
            Candidate(
                smiles=can,
                method=method,
                seed_name=seed.name,
                valid=True,
                novel=novel,
                tanimoto=sim,
                qed=qed,
                sa=sa,
                delta_qed=qed - float(seed.qed),
                kept=kept,
            )
        )
    return result


def summarize(result: MethodResult) -> dict:
    valid = result.valid
    kept = result.kept_cands
    n_valid = len(valid)
    n_unique = len({c.smiles for c in valid})
    n_novel = len({c.smiles for c in valid if c.novel})
    kept_sims = [c.tanimoto for c in kept if c.tanimoto is not None]
    kept_dq = [c.delta_qed for c in kept if c.delta_qed is not None]
    kept_qed = [c.qed for c in kept if c.qed is not None]
    kept_sa = [c.sa for c in kept if c.sa is not None]
    row = {
        "seed": result.seed_name,
        "method": result.method,
        "seed_qed": result.seed_qed,
        "attempted": result.attempted,
        "n_valid": n_valid,
        "validity": n_valid / result.attempted if result.attempted else 0.0,
        "uniqueness": n_unique / n_valid if n_valid else 0.0,
        "novelty": n_novel / n_valid if n_valid else 0.0,
        "n_kept": len(kept),
        "mean_tanimoto_kept": float(np.mean(kept_sims)) if kept_sims else np.nan,
        "max_tanimoto_kept": float(np.max(kept_sims)) if kept_sims else np.nan,
        "mean_qed_kept": float(np.mean(kept_qed)) if kept_qed else np.nan,
        "mean_delta_qed": float(np.mean(kept_dq)) if kept_dq else np.nan,
        "max_delta_qed": float(np.max(kept_dq)) if kept_dq else np.nan,
        "frac_improved": float(np.mean([d > 0 for d in kept_dq])) if kept_dq else np.nan,
        "mean_sa_kept": float(np.mean(kept_sa)) if kept_sa else np.nan,
    }
    return row


def results_table(results: list[MethodResult]) -> pd.DataFrame:
    return pd.DataFrame([summarize(r) for r in results])


def aggregate_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "validity",
        "uniqueness",
        "novelty",
        "n_kept",
        "mean_tanimoto_kept",
        "mean_delta_qed",
        "max_delta_qed",
        "frac_improved",
        "mean_sa_kept",
    ]
    return df.groupby("method")[metrics].mean(numeric_only=True).reset_index()


def top_kept(result: MethodResult, k: int = 3) -> list[Candidate]:
    kept = [c for c in result.kept_cands]
    kept.sort(key=lambda c: (c.delta_qed is not None, c.delta_qed), reverse=True)
    # unique smiles already enforced in kept
    out = []
    seen = set()
    for c in kept:
        if c.smiles in seen:
            continue
        seen.add(c.smiles)
        out.append(c)
        if len(out) >= k:
            break
    return out
