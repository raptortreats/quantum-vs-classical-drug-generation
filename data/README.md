# Public reference structures

This folder holds a **tiny curated public SMILES set** used as:

1. Seed molecules for the goal-directed bake-off (`seeds.csv`)
2. A drug-like reference / training corpus (`reference_smiles.tsv`)

## Provenance

- Primary source: [`HIPS/molecule-autoencoder`](https://github.com/HIPS/molecule-autoencoder) file `data/all_drugs.smi` (public GitHub).
- That list contains widely documented small-molecule drugs (INN / FDA-approved structures) distributed for research on chemical autoencoders.
- Related scientific context: Gómez-Bombarelli et al., *Automatic chemical design using a data-driven continuous representation of molecules*, ACS Cent. Sci. 2018 (and the HIPS autoencoder code release).
- **No proprietary screening collections, no commercial vendor libraries, no private pharma data.**

## Filtering applied here

From the public list we keep organic, single-fragment, charge-neutral molecules with roughly drug-like size (8–36 heavy atoms, MW 120–550 Da, canonical SMILES length compatible with the GRU). Salts, disconnected mixtures, and parse failures are dropped. Canonical SMILES are regenerated with RDKit.

Seeds are six well-known public drugs spanning low → high QED so the bake-off is not trivially “everything already maxed out.”
