"""Plotting helpers for the bake-off notebook."""

from __future__ import annotations

from pathlib import Path

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw
from rdkit.Chem.Draw import rdMolDraw2D

from .chemistry import mol_from_smiles
from .config import FIGURE_DIR
from .data import MoleculeRecord
from .metrics import MethodResult, top_kept

CLASSICAL_COLOR = "#2C6EAC"
HYBRID_COLOR = "#7B3FA0"
METHOD_PALETTE = {"classical_gru": CLASSICAL_COLOR, "hybrid_qcbm": HYBRID_COLOR}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["savefig.bbox"] = "tight"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _grid_to_pil(mols, legends, mols_per_row: int, sub_img_size=(160, 140)) -> Image.Image:
    mols = list(mols)
    legends = list(legends)
    n = max(len(mols), 1)
    ncols = max(mols_per_row, 1)
    nrows = int(np.ceil(n / ncols))
    w, h = sub_img_size
    drawer = rdMolDraw2D.MolDraw2DCairo(ncols * w, nrows * h, w, h)
    drawer.DrawMolecules(mols, legends=legends)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def savefig(fig: plt.Figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path)
    return path


def plot_gru_loss(losses: list[float], extra: dict[str, list[float]] | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(np.arange(1, len(losses) + 1), losses, color=CLASSICAL_COLOR, lw=2, label="pretrain")
    if extra:
        for seed_name, vals in extra.items():
            ax.plot(np.arange(1, len(vals) + 1), vals, lw=1.4, alpha=0.85, label=f"ft {seed_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Token NLL")
    ax.set_title("Classical SELFIES GRU training")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_qcbm_losses(loss_map: dict[str, list[float]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (name, losses) in enumerate(loss_map.items()):
        ax.plot(np.arange(1, len(losses) + 1), losses, lw=1.8, label=name)
    ax.set_xlabel("Adam step")
    ax.set_ylabel("KL(target ‖ QCBM)")
    ax.set_title("Hybrid QCBM fit to QED-weighted neighborhoods")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_metric_bars(df: pd.DataFrame) -> plt.Figure:
    metrics = ["validity", "uniqueness", "novelty", "frac_improved"]
    long = df.melt(id_vars=["seed", "method"], value_vars=metrics, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    sns.barplot(
        data=long,
        x="metric",
        y="value",
        hue="method",
        palette=METHOD_PALETTE,
        ax=ax,
        errorbar="sd",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Generation quality (mean ± sd across seeds)")
    ax.legend(title="", frameon=False)
    fig.tight_layout()
    return fig


def plot_delta_qed(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    sns.barplot(
        data=df,
        x="seed",
        y="mean_delta_qed",
        hue="method",
        palette=METHOD_PALETTE,
        ax=ax,
    )
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("Mean ΔQED among kept")
    ax.set_xlabel("")
    ax.set_title("Goal-directed improvement vs seed (kept analogs)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="", frameon=False)
    fig.tight_layout()
    return fig


def plot_scatter(results: list[MethodResult]) -> plt.Figure:
    rows = []
    for res in results:
        for c in res.kept_cands:
            rows.append(
                {
                    "method": res.method,
                    "seed": res.seed_name,
                    "tanimoto": c.tanimoto,
                    "delta_qed": c.delta_qed,
                }
            )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    if df.empty:
        ax.text(0.5, 0.5, "No kept candidates", ha="center")
        return fig
    sns.scatterplot(
        data=df,
        x="tanimoto",
        y="delta_qed",
        hue="method",
        style="seed",
        palette=METHOD_PALETTE,
        ax=ax,
        s=42,
        alpha=0.85,
    )
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel("Tanimoto similarity to seed")
    ax.set_ylabel("ΔQED vs seed")
    ax.set_title("Kept candidates: similarity vs drug-likeness gain")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def molecule_grid(
    seed: MoleculeRecord,
    classical: MethodResult,
    hybrid: MethodResult,
    k: int = 3,
) -> Image.Image:
    """Stack seed / classical / hybrid rows so legends stay readable."""

    def _row(title: str, recs: list) -> Image.Image:
        if not recs:
            img = Image.new("RGB", (k * 220, 88), "white")
            ImageDraw.Draw(img).text(
                (12, 36),
                f"{title}: no kept analogs at Tanimoto >= 0.40",
                fill=(80, 80, 80),
            )
            return img
        mols = [mol_from_smiles(c.smiles) for c in recs]
        legends = [
            f"{title} #{i}\nΔQED {c.delta_qed:+.3f}  Tc {c.tanimoto:.2f}\nQED {c.qed:.3f}"
            for i, c in enumerate(recs, 1)
        ]
        return _grid_to_pil(mols, legends, mols_per_row=max(len(mols), 1), sub_img_size=(220, 200))

    seed_img = _grid_to_pil(
        [seed.mol],
        [f"SEED  {seed.name}\nQED {seed.qed:.3f}  SA {seed.sa:.2f}"],
        mols_per_row=1,
        sub_img_size=(240, 210),
    )
    c_img = _row("classical", top_kept(classical, k=k))
    h_img = _row("hybrid", top_kept(hybrid, k=k))
    width = max(seed_img.width, c_img.width, h_img.width)
    gap = 8
    height = seed_img.height + c_img.height + h_img.height + 2 * gap
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for part in (seed_img, c_img, h_img):
        canvas.paste(part, (0, y))
        y += part.height + gap
    return canvas


def save_seed_grid(seeds: list[MoleculeRecord]) -> Path:
    img = _grid_to_pil(
        [s.mol for s in seeds],
        [f"{s.name}\nQED {s.qed:.3f}  SA {s.sa:.2f}" for s in seeds],
        mols_per_row=3,
        sub_img_size=(200, 160),
    )
    path = FIGURE_DIR / "seeds.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path
