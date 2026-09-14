"""Load the public reference corpus and bake-off seeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from rdkit import Chem

from .chemistry import canonical_smiles, mol_from_smiles, qed_score, sa_score
from .config import DATA_DIR, GRU_MAX_LEN


@dataclass
class MoleculeRecord:
    name: str
    smiles: str
    source: str
    qed: Optional[float] = None
    sa: Optional[float] = None
    mol: Optional[Chem.Mol] = None


def _annotate(name: str, smiles: str, source: str) -> Optional[MoleculeRecord]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    can = canonical_smiles(mol)
    if can is None or "." in can:
        return None
    if len(can) > GRU_MAX_LEN + 10:
        return None
    return MoleculeRecord(
        name=name,
        smiles=can,
        source=source,
        qed=qed_score(mol),
        sa=sa_score(mol),
        mol=mol,
    )


def load_seeds() -> list[MoleculeRecord]:
    path = DATA_DIR / "seeds.csv"
    df = pd.read_csv(path)
    records = []
    for row in df.itertuples(index=False):
        rec = _annotate(str(row.name), str(row.smiles), str(row.source))
        if rec is not None:
            records.append(rec)
    if not records:
        raise RuntimeError("No valid seed molecules found in data/seeds.csv")
    return records


def load_reference() -> list[MoleculeRecord]:
    path = DATA_DIR / "reference_smiles.tsv"
    df = pd.read_csv(path, sep="\t")
    records = []
    seen = set()
    for row in df.itertuples(index=False):
        rec = _annotate(str(row.name), str(row.smiles), str(row.source))
        if rec is None or rec.smiles in seen:
            continue
        seen.add(rec.smiles)
        records.append(rec)
    if len(records) < 50:
        raise RuntimeError("Reference corpus is unexpectedly small")
    return records


def reference_smiles_set(records: list[MoleculeRecord]) -> set[str]:
    return {r.smiles for r in records}
