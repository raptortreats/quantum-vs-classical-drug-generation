"""Chemistry utilities: validity, fingerprints, QED, SA, analog mutations."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Callable, Iterable, Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, RDConfig, rdFingerprintGenerator
from rdkit.Chem.Descriptors import BertzCT, MolWt

from .config import MORGAN_NBITS, MORGAN_RADIUS

RDLogger.DisableLog("rdApp.*")

FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS)

try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # type: ignore

    def sa_score(mol: Chem.Mol) -> float:
        return float(sascorer.calculateScore(mol))

    SA_AVAILABLE = True
except Exception:  # pragma: no cover - contrib path varies by install
    def sa_score(mol: Chem.Mol) -> float:
        """Fallback: scaled Bertz complexity (higher = more complex)."""
        return float(BertzCT(mol) / 50.0)

    SA_AVAILABLE = False


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if mol.GetNumHeavyAtoms() < 3:
        return None
    return mol


def canonical_smiles(mol_or_smi) -> Optional[str]:
    mol = mol_or_smi if isinstance(mol_or_smi, Chem.Mol) else mol_from_smiles(mol_or_smi)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    return mol_from_smiles(smiles) is not None


def qed_score(mol: Chem.Mol) -> float:
    return float(QED.qed(mol))


def mol_wt(mol: Chem.Mol) -> float:
    return float(MolWt(mol))


def morgan_fp(mol: Chem.Mol):
    return FPGEN.GetFingerprint(mol)


def fp_to_numpy(fp) -> np.ndarray:
    arr = np.zeros((fp.GetNumBits(),), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto(fp_a, fp_b) -> float:
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def tanimoto_smiles(seed_smi: str, other_smi: str) -> Optional[float]:
    a = mol_from_smiles(seed_smi)
    b = mol_from_smiles(other_smi)
    if a is None or b is None:
        return None
    return tanimoto(morgan_fp(a), morgan_fp(b))


def randomized_smiles(mol: Chem.Mol, n: int, rng: np.random.Generator) -> list[str]:
    out = []
    seen = set()
    can = Chem.MolToSmiles(mol, canonical=True)
    out.append(can)
    seen.add(can)
    for _ in range(n * 4):
        if len(out) >= n + 1:
            break
        try:
            smi = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        except Exception:
            continue
        if smi and smi not in seen:
            seen.add(smi)
            out.append(smi)
    return out


def _safe_mol(rw: Chem.RWMol) -> Optional[Chem.Mol]:
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        if mol.GetNumHeavyAtoms() < 3:
            return None
        Chem.MolToSmiles(mol, canonical=True)
        return mol
    except Exception:
        return None


def _copy(mol: Chem.Mol) -> Chem.RWMol:
    return Chem.RWMol(Chem.Mol(mol))


def op_add_methyl(mol: Chem.Mol, site: int) -> Optional[Chem.Mol]:
    rw = _copy(mol)
    cands = [
        a.GetIdx()
        for a in rw.GetAtoms()
        if a.GetAtomicNum() == 6 and a.GetTotalNumHs() >= 1 and a.GetDegree() < 4
    ]
    if not cands:
        return None
    idx = cands[site % len(cands)]
    new = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(idx, new, Chem.BondType.SINGLE)
    return _safe_mol(rw)


def op_add_fluoro(mol: Chem.Mol, site: int) -> Optional[Chem.Mol]:
    rw = _copy(mol)
    cands = [
        a.GetIdx()
        for a in rw.GetAtoms()
        if a.GetIsAromatic() and a.GetAtomicNum() == 6 and a.GetTotalNumHs() >= 1
    ]
    if not cands:
        return None
    idx = cands[site % len(cands)]
    new = rw.AddAtom(Chem.Atom(9))
    rw.AddBond(idx, new, Chem.BondType.SINGLE)
    return _safe_mol(rw)


def op_add_hydroxy(mol: Chem.Mol, site: int) -> Optional[Chem.Mol]:
    rw = _copy(mol)
    cands = [
        a.GetIdx()
        for a in rw.GetAtoms()
        if a.GetAtomicNum() == 6 and a.GetTotalNumHs() >= 1
    ]
    if not cands:
        return None
    idx = cands[site % len(cands)]
    new = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(idx, new, Chem.BondType.SINGLE)
    return _safe_mol(rw)


def op_aromatic_c_to_n(mol: Chem.Mol, site: int) -> Optional[Chem.Mol]:
    rw = _copy(mol)
    cands = [
        a.GetIdx()
        for a in rw.GetAtoms()
        if a.GetIsAromatic() and a.GetAtomicNum() == 6 and a.GetTotalNumHs() >= 1
    ]
    if not cands:
        return None
    idx = cands[site % len(cands)]
    rw.GetAtomWithIdx(idx).SetAtomicNum(7)
    return _safe_mol(rw)


def op_aromatic_n_to_c(mol: Chem.Mol, site: int) -> Optional[Chem.Mol]:
    rw = _copy(mol)
    cands = [
        a.GetIdx()
        for a in rw.GetAtoms()
        if a.GetIsAromatic() and a.GetAtomicNum() == 7
    ]
    if not cands:
        return None
    idx = cands[site % len(cands)]
    rw.GetAtomWithIdx(idx).SetAtomicNum(6)
    return _safe_mol(rw)


def op_oh_to_ome(mol: Chem.Mol, site: int) -> Optional[Chem.Mol]:
    rw = _copy(mol)
    cands = []
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and a.GetTotalNumHs() >= 1:
            nbr = a.GetNeighbors()[0]
            if nbr.GetAtomicNum() == 6:
                cands.append(a.GetIdx())
    if not cands:
        return None
    idx = cands[site % len(cands)]
    new = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(idx, new, Chem.BondType.SINGLE)
    return _safe_mol(rw)


def op_ome_to_oh(mol: Chem.Mol, site: int) -> Optional[Chem.Mol]:
    rw = _copy(mol)
    methyl = []
    for a in rw.GetAtoms():
        if a.GetAtomicNum() != 6 or a.GetDegree() != 1:
            continue
        nbrs = a.GetNeighbors()
        if nbrs and nbrs[0].GetAtomicNum() == 8:
            methyl.append(a.GetIdx())
    if not methyl:
        return None
    idx = methyl[site % len(methyl)]
    rw.RemoveAtom(idx)
    return _safe_mol(rw)


def op_acid_to_amide(mol: Chem.Mol, site: int) -> Optional[Chem.Mol]:
    """COOH carbon -> CONH2 by adding N on the carbonyl after finding acid OH."""
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)[OH]"))
    if not matches:
        return None
    acid_c, _, _oh = matches[site % len(matches)]
    rw = _copy(mol)
    # replace OH oxygen with NH2 nitrogen
    rw.GetAtomWithIdx(_oh).SetAtomicNum(7)
    return _safe_mol(rw)


def op_remove_leaf_methyl(mol: Chem.Mol, site: int) -> Optional[Chem.Mol]:
    rw = _copy(mol)
    cands = []
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 6 and a.GetDegree() == 1 and not a.GetIsAromatic():
            nbr = a.GetNeighbors()[0]
            if nbr.GetAtomicNum() == 6:
                cands.append(a.GetIdx())
    if not cands:
        return None
    idx = cands[site % len(cands)]
    rw.RemoveAtom(idx)
    return _safe_mol(rw)


MUTATION_OPS: list[tuple[str, Callable[[Chem.Mol, int], Optional[Chem.Mol]]]] = [
    ("add_methyl", op_add_methyl),
    ("add_fluoro", op_add_fluoro),
    ("add_hydroxy", op_add_hydroxy),
    ("aromatic_c_to_n", op_aromatic_c_to_n),
    ("aromatic_n_to_c", op_aromatic_n_to_c),
    ("oh_to_ome", op_oh_to_ome),
    ("ome_to_oh", op_ome_to_oh),
    ("acid_to_amide", op_acid_to_amide),
    ("remove_leaf_methyl", op_remove_leaf_methyl),
]


def mutate_from_bits(mol: Chem.Mol, bitstring: int, n_tries: int = 4) -> Optional[Chem.Mol]:
    """Apply a bitstring-conditioned analog mutation; try nearby ops if one fails."""
    parent = canonical_smiles(mol)
    n_ops = len(MUTATION_OPS)
    for k in range(n_tries):
        op_idx = (bitstring + k) % n_ops
        site = (bitstring // n_ops + k * 3) % 17
        name, fn = MUTATION_OPS[op_idx]
        child = fn(mol, site)
        if child is None:
            continue
        child_smi = canonical_smiles(child)
        if child_smi and child_smi != parent:
            return child
    return None


def enumerate_analogs(mol: Chem.Mol, limit: int = 24) -> list[Chem.Mol]:
    """Deterministic small analog cloud around a parent (classical local moves)."""
    parent = canonical_smiles(mol)
    out: list[Chem.Mol] = []
    seen = {parent}
    for op_idx, (_, fn) in enumerate(MUTATION_OPS):
        for site in range(8):
            child = fn(mol, site)
            if child is None:
                continue
            smi = canonical_smiles(child)
            if smi and smi not in seen:
                seen.add(smi)
                out.append(child)
                if len(out) >= limit:
                    return out
    return out


@lru_cache(maxsize=1)
def _sa_flag() -> bool:
    return SA_AVAILABLE
