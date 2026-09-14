"""SELFIES-token GRU generator with seed-prefix sampling.

SELFIES (Krenn et al., 2020) makes the language-model baseline chemically
robust: almost every sampled string decodes to a valid molecule, so the
bake-off can compare *goal-directed quality* rather than SMILES syntax.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import selfies as sf
import torch
import torch.nn as nn
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset

from .chemistry import canonical_smiles, mol_from_smiles, randomized_smiles
from .config import (
    GRU_BATCH_SIZE,
    GRU_DROPOUT,
    GRU_EMBED_DIM,
    GRU_EPOCHS,
    GRU_HIDDEN_DIM,
    GRU_LAYERS,
    GRU_LR,
    GRU_MAX_LEN,
    GRU_TEMPERATURE,
    RANDOM_SEED,
)
from .data import MoleculeRecord

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


def smiles_to_selfies(smiles: str) -> str | None:
    try:
        return sf.encoder(smiles)
    except Exception:
        try:
            mol = mol_from_smiles(smiles)
            if mol is None:
                return None
            kek = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)
            return sf.encoder(kek)
        except Exception:
            return None


def selfies_to_smiles(selfies: str) -> str | None:
    try:
        smi = sf.decoder(selfies)
    except Exception:
        return None
    return canonical_smiles(smi)


class SelfiesVocab:
    def __init__(self, tokens: list[str]):
        special = [PAD, BOS, EOS, UNK]
        uniq = special + sorted(set(tokens) - set(special))
        self.stoi = {t: i for i, t in enumerate(uniq)}
        self.itos = {i: t for t, i in self.stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, selfies: str, max_len: int = GRU_MAX_LEN) -> list[int]:
        toks = [BOS] + list(sf.split_selfies(selfies))[: max_len - 2] + [EOS]
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in toks]
        ids += [self.stoi[PAD]] * (max_len - len(ids))
        return ids[:max_len]

    def prefix_ids(self, selfies: str, n_tokens: int) -> list[int]:
        toks = [BOS] + list(sf.split_selfies(selfies))[:n_tokens]
        return [self.stoi.get(t, self.stoi[UNK]) for t in toks]

    def decode_selfies(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            tok = self.itos.get(int(i), UNK)
            if tok in (BOS, PAD, UNK):
                continue
            if tok == EOS:
                break
            out.append(tok)
        return "".join(out)


class SelfiesDataset(Dataset):
    def __init__(self, sequences: list[list[int]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        return x[:-1], x[1:]


class SmilesGRU(nn.Module):
    """Token GRU; trained on SELFIES rather than raw SMILES characters."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, GRU_EMBED_DIM, padding_idx=0)
        self.gru = nn.GRU(
            GRU_EMBED_DIM,
            GRU_HIDDEN_DIM,
            GRU_LAYERS,
            batch_first=True,
            dropout=GRU_DROPOUT if GRU_LAYERS > 1 else 0.0,
        )
        self.head = nn.Linear(GRU_HIDDEN_DIM, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        y, hidden = self.gru(emb, hidden)
        return self.head(y), hidden


@dataclass
class ClassicalTrainer:
    vocab: SelfiesVocab
    model: SmilesGRU
    device: torch.device
    losses: list[float]


def _device() -> torch.device:
    return torch.device("cpu")


def build_training_selfies(
    records: list[MoleculeRecord], n_random: int, rng: np.random.Generator
) -> list[str]:
    strings = []
    for rec in records:
        mol = rec.mol if rec.mol is not None else mol_from_smiles(rec.smiles)
        if mol is None:
            continue
        for smi in randomized_smiles(mol, n_random, rng):
            sfs = smiles_to_selfies(smi)
            if not sfs:
                continue
            n_tok = len(list(sf.split_selfies(sfs)))
            if 3 <= n_tok <= GRU_MAX_LEN - 2:
                strings.append(sfs)
    return strings


def train_gru(
    records: list[MoleculeRecord],
    n_random: int = 4,
    epochs: int = GRU_EPOCHS,
    seed: int = RANDOM_SEED,
) -> ClassicalTrainer:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    strings = build_training_selfies(records, n_random, rng)
    tokens: list[str] = []
    for s in strings:
        tokens.extend(sf.split_selfies(s))
    vocab = SelfiesVocab(tokens)
    sequences = [vocab.encode(s) for s in strings]
    loader = DataLoader(
        SelfiesDataset(sequences),
        batch_size=GRU_BATCH_SIZE,
        shuffle=True,
        drop_last=len(sequences) > GRU_BATCH_SIZE,
    )
    device = _device()
    model = SmilesGRU(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=GRU_LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.stoi[PAD])
    losses = []
    model.train()
    for _ in range(epochs):
        running = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            running.append(float(loss.item()))
        losses.append(float(np.mean(running)) if running else float("nan"))
    return ClassicalTrainer(vocab=vocab, model=model, device=device, losses=losses)


def finetune_gru(
    trainer: ClassicalTrainer,
    neighborhood: list[MoleculeRecord],
    epochs: int,
    n_random: int = 6,
    seed: int = RANDOM_SEED,
) -> list[float]:
    if not neighborhood:
        return []
    rng = np.random.default_rng(seed + 7)
    strings = build_training_selfies(neighborhood, n_random, rng)
    if not strings:
        return []
    sequences = [trainer.vocab.encode(s) for s in strings]
    loader = DataLoader(
        SelfiesDataset(sequences),
        batch_size=min(32, len(sequences)),
        shuffle=True,
    )
    opt = torch.optim.Adam(trainer.model.parameters(), lr=GRU_LR * 0.5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=trainer.vocab.stoi[PAD])
    trainer.model.train()
    losses = []
    for _ in range(epochs):
        running = []
        for x, y in loader:
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            logits, _ = trainer.model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running.append(float(loss.item()))
        losses.append(float(np.mean(running)) if running else float("nan"))
    return losses


@torch.no_grad()
def generate_smiles(
    trainer: ClassicalTrainer,
    n: int,
    rng: np.random.Generator,
    seed_smiles: str | None = None,
    temperature: float = GRU_TEMPERATURE,
    max_len: int = GRU_MAX_LEN,
) -> list[str]:
    """Sample SELFIES (optionally seed-prefixed) and decode to canonical SMILES."""
    model = trainer.model
    vocab = trainer.vocab
    model.eval()
    bos = vocab.stoi[BOS]
    eos = vocab.stoi[EOS]
    pad = vocab.stoi[PAD]
    unk = vocab.stoi[UNK]
    out = []
    seed_sf = smiles_to_selfies(seed_smiles) if seed_smiles else None
    seed_toks = list(sf.split_selfies(seed_sf)) if seed_sf else []
    for _ in range(n):
        if seed_toks and rng.random() < 0.88:
            # lock most of the seed SELFIES so completions stay in-neighborhood
            if rng.random() < 0.35:
                frac = float(rng.uniform(0.78, 0.93))
            else:
                frac = float(rng.uniform(0.50, 0.78))
            n_pref = max(3, int(len(seed_toks) * frac))
            ids = vocab.prefix_ids(seed_sf, n_pref)
        else:
            ids = [bos]
        x = torch.tensor(ids, dtype=torch.long, device=trainer.device).unsqueeze(0)
        hidden = None
        logits, hidden = model(x, hidden)
        next_logits = logits[:, -1, :]
        generated = list(ids)
        for _step in range(max_len - len(generated)):
            logits_t = next_logits.squeeze(0) / max(temperature, 1e-4)
            logits_t[pad] = -1e9
            logits_t[bos] = -1e9
            logits_t[unk] = -1e9
            probs = torch.softmax(logits_t, dim=-1).cpu().numpy()
            probs = np.clip(probs, 0, None)
            z = probs.sum()
            if z <= 0:
                break
            if rng.random() < 0.1:
                nxt = int(np.argmax(probs))
            else:
                nxt = int(rng.choice(len(probs), p=probs / z))
            generated.append(nxt)
            if nxt == eos:
                break
            token = torch.tensor([[nxt]], dtype=torch.long, device=trainer.device)
            next_logits, hidden = model(token, hidden)
            next_logits = next_logits[:, -1, :]
        sfs = vocab.decode_selfies(generated)
        smi = selfies_to_smiles(sfs)
        out.append(smi if smi else "")
    return out
