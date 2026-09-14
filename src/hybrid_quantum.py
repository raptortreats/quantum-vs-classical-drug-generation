"""Hybrid quantum-classical generator: QCBM on fingerprint latents + analog decode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from sklearn.decomposition import PCA

from .chemistry import (
    canonical_smiles,
    enumerate_analogs,
    fp_to_numpy,
    mol_from_smiles,
    morgan_fp,
    mutate_from_bits,
    qed_score,
    tanimoto,
)
from .config import (
    N_QUBITS,
    QCBM_LAYERS,
    QCBM_LR,
    QCBM_NEIGHBOR_K,
    QCBM_QED_TEMP,
    QCBM_STEPS,
    RANDOM_SEED,
)
from .data import MoleculeRecord


@dataclass
class BitEncoder:
    pca: PCA
    thresholds: np.ndarray

    def transform(self, fp_matrix: np.ndarray) -> np.ndarray:
        z = self.pca.transform(fp_matrix)
        return (z > self.thresholds).astype(np.int32)

    def transform_fp(self, fp) -> int:
        bits = self.transform(fp_to_numpy(fp).reshape(1, -1))[0]
        return int(bits_to_int(bits))


def bits_to_int(bits: np.ndarray) -> int:
    acc = 0
    for b in bits:
        acc = (acc << 1) | int(b)
    return acc


def int_to_bits(value: int, n: int = N_QUBITS) -> np.ndarray:
    return np.array([(value >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.int32)


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def fit_bit_encoder(records: list[MoleculeRecord], n_bits: int = N_QUBITS, seed: int = RANDOM_SEED) -> BitEncoder:
    X = np.stack([fp_to_numpy(morgan_fp(r.mol)) for r in records])
    pca = PCA(n_components=n_bits, random_state=seed)
    z = pca.fit_transform(X)
    thresholds = np.median(z, axis=0)
    return BitEncoder(pca=pca, thresholds=thresholds)


def neighborhood(
    seed: MoleculeRecord,
    library: list[MoleculeRecord],
    k: int = QCBM_NEIGHBOR_K,
) -> list[tuple[MoleculeRecord, float]]:
    seed_fp = morgan_fp(seed.mol)
    scored = []
    for rec in library:
        if rec.smiles == seed.smiles:
            continue
        sim = tanimoto(seed_fp, morgan_fp(rec.mol))
        scored.append((rec, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def target_distribution(
    neighbors: list[tuple[MoleculeRecord, float]],
    encoder: BitEncoder,
    seed_qed: float,
    n_qubits: int = N_QUBITS,
) -> np.ndarray:
    dim = 2 ** n_qubits
    weights = np.zeros(dim, dtype=np.float64)
    for rec, sim in neighbors:
        bits = encoder.transform_fp(morgan_fp(rec.mol))
        w = np.exp(QCBM_QED_TEMP * (float(rec.qed) - seed_qed)) * (0.5 + sim)
        weights[bits] += w
    # always give the seed itself a small mass so the latent stays local
    weights = np.clip(weights, 0, None)
    if weights.sum() <= 0:
        weights[:] = 1.0
    return weights / weights.sum()


def make_qcbm(n_qubits: int = N_QUBITS, n_layers: int = QCBM_LAYERS):
    """Explicit strongly-entangling ansatz (Rot + ring of CNOTs)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def probs(params):
        for layer in range(n_layers):
            for wire in range(n_qubits):
                qml.Rot(
                    params[layer, wire, 0],
                    params[layer, wire, 1],
                    params[layer, wire, 2],
                    wires=wire,
                )
            for wire in range(n_qubits):
                qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
        return qml.probs(wires=range(n_qubits))

    return probs


def random_params(n_qubits: int = N_QUBITS, n_layers: int = QCBM_LAYERS, seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.3, size=(n_layers, n_qubits, 3))


def train_qcbm(
    target: np.ndarray,
    steps: int = QCBM_STEPS,
    lr: float = QCBM_LR,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, list[float], callable]:
    n_qubits = int(np.log2(len(target)))
    circuit = make_qcbm(n_qubits=n_qubits)
    params = pnp.array(random_params(n_qubits=n_qubits, seed=seed), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)
    tgt = np.clip(target, 1e-8, 1.0)
    tgt = tgt / tgt.sum()
    tgt = pnp.array(tgt)

    def loss_fn(p):
        pr = circuit(p)
        pr = pr + 1e-8
        pr = pr / qml.math.sum(pr)
        # KL(target || model) — fit the QED-weighted neighborhood
        return qml.math.sum(tgt * (qml.math.log(tgt) - qml.math.log(pr)))

    losses = []
    for _ in range(steps):
        params, loss = opt.step_and_cost(loss_fn, params)
        losses.append(float(loss))
    return params, losses, circuit


def sample_bitstrings(circuit, params, n: int, rng: np.random.Generator) -> np.ndarray:
    pr = np.array(circuit(params), dtype=np.float64)
    pr = np.clip(pr, 0, None)
    pr = pr / pr.sum()
    return rng.choice(len(pr), size=n, p=pr)


def _pick_parent(
    bitstring: int,
    neighbors: list[tuple[MoleculeRecord, float]],
    encoder: BitEncoder,
    seed: MoleculeRecord,
    rng: np.random.Generator,
) -> MoleculeRecord:
    if not neighbors:
        return seed
    ranked = []
    for rec, sim in neighbors:
        code = encoder.transform_fp(morgan_fp(rec.mol))
        ranked.append((hamming(bitstring, code), -sim, -float(rec.qed), rec))
    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    top = [r[3] for r in ranked[: min(5, len(ranked))]]
    # mix seed to keep structural similarity high
    if rng.random() < 0.45:
        return seed
    return top[int(rng.integers(0, len(top)))]


def decode_candidates(
    bitstrings: np.ndarray,
    seed: MoleculeRecord,
    neighbors: list[tuple[MoleculeRecord, float]],
    encoder: BitEncoder,
    rng: np.random.Generator,
) -> list[str]:
    smiles_out = []
    analog_cache: dict[str, list] = {}
    for b in bitstrings:
        b = int(b)
        parent = _pick_parent(b, neighbors, encoder, seed, rng)
        child = mutate_from_bits(parent.mol, b)
        if child is None:
            # fallback: enumerate a small analog cloud once per parent
            if parent.smiles not in analog_cache:
                analog_cache[parent.smiles] = enumerate_analogs(parent.mol, limit=16)
            cloud = analog_cache[parent.smiles]
            if cloud:
                child = cloud[b % len(cloud)]
        if child is None:
            smiles_out.append("")
            continue
        smi = canonical_smiles(child) or ""
        smiles_out.append(smi)
    return smiles_out


def generate_hybrid(
    seed: MoleculeRecord,
    library: list[MoleculeRecord],
    encoder: BitEncoder,
    n: int,
    rng: np.random.Generator,
    steps: int = QCBM_STEPS,
) -> tuple[list[str], list[float], np.ndarray]:
    neigh = neighborhood(seed, library, k=QCBM_NEIGHBOR_K)
    target = target_distribution(neigh, encoder, seed_qed=float(seed.qed))
    params, losses, circuit = train_qcbm(target, steps=steps, seed=int(rng.integers(1, 10_000)))
    bits = sample_bitstrings(circuit, params, n=n, rng=rng)
    smiles = decode_candidates(bits, seed, neigh, encoder, rng)
    return smiles, losses, params
