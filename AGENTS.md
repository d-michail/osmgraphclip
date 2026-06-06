# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository. For a project overview, pretrained models, and training dataset information see `README.md`.

## Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build a dataset

Dataset creation is a two-step process:

1. **`create_dataset.py`** — downloads OSM GeoJSON files and populates the `downloads` table in `dataset.db`
2. **`create_graphs.py`** — builds graph pickles from the GeoJSON files and populates the `graphs` table in `dataset.db`

Both steps are independently resumable.

#### `create_dataset.py` — batch download

```bash
# From a CSV of locations
python create_dataset.py \
  --output-dir location_dataset \
  --locations-file data/locations.csv \
  --bbox-size 250

# Built-in city grid
python create_dataset.py --output-dir athens --city athens

# Adaptive bbox with profile
python create_dataset.py --output-dir dataset --locations-file data/locations.csv \
  --adaptive-bbox --bbox-profile rural

# Adaptive bbox with manual size control
python create_dataset.py --output-dir athens --city athens \
  --adaptive-bbox --bbox-size 250 --max-bbox-size 1000 \
  --bbox-expansion-strategy multiplicative --bbox-expansion-factor 1.5 \
  --min-total-nodes 5 --min-unique-labels 3 --min-richness-score 0.2

# Force full re-download
python create_dataset.py ... --no-resume
```

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | required | Directory to store the dataset |
| `--locations-file` | — | CSV with `lat`/`lon` columns (mutually exclusive with `--city`) |
| `--city` | — | Built-in city name to generate a grid dataset |
| `--sample-spacing` | `500` | Grid spacing in metres (only with `--city`) |
| `--bbox-size` | `250` | Initial bbox half-width in metres; overrides `--bbox-profile` |
| `--bbox-profile` | — | Preset: `dense_city`, `suburb`, `rural`, `wilderness`, `regional` |
| `--tagw-path` | `data/all_tags30_frequency1.json` | Tag weights JSON (needed for richness scoring with `--adaptive-bbox`) |
| `--adaptive-bbox` | off | Expand bbox until richness thresholds are met |
| `--max-bbox-size` | — | Absolute bbox ceiling in metres (overrides `--adaptive-max-factor`) |
| `--adaptive-max-factor` | `4.0` | Ceiling = initial × factor (used when `--max-bbox-size` is not set) |
| `--bbox-expansion-strategy` | `multiplicative` | `multiplicative` or `linear` |
| `--bbox-expansion-factor` | `1.5` | Growth factor per retry (multiplicative strategy) |
| `--bbox-expansion-step` | `100` | Step in metres per retry (linear strategy) |
| `--min-total-nodes` | `5` | Minimum OSM feature count |
| `--min-unique-labels` | `3` | Minimum distinct OSM tag labels |
| `--min-richness-score` | `0.2` | Minimum composite richness score [0, 1] |
| `--multiresolution` | off | Download at multiple bbox levels per location |
| `--levels` | `200 500 2000 5000 20000` | Bbox sizes in metres for each level (fine → coarse) |
| `--best-level-only` | off | Keep only the richest level per location (requires `--multiresolution`) |
| `--best-level-richness-weight` | `1.0` | Weight for composite richness in best-level scoring |
| `--best-level-depth-weight` | `0.2` | Weight for semantic depth (avg non-null tags per feature) |
| `--best-level-categories-weight` | `0.1` | Weight for category coverage |
| `--best-level-size-weight` | `0.1` | Penalty for bbox size (prefers finer levels) |
| `--best-level-node-weight` | `0.5` | Penalty for node count (targets ~20–30 nodes) |
| `--best-level-min-nodes` | `10` | Levels below this node count receive a large penalty |
| `--save-road-graph` | off | Also save OSM road graph as GraphML |
| `--backend` | `overpass` | Data source: `overpass`, `postgis`, or `auto` (PostGIS → Overpass fallback) |
| `--postgis-url` | — | PostgreSQL DSN for PostGIS backend (or `POSTGIS_URL` env var) |
| `--postgis-max-rows` | `50000` | Max rows per geometry table from PostGIS; 0 = unlimited |
| `--nodata-retry-after` | — | Retry `.nodata` sentinel files older than N hours |
| `--resume` / `--no-resume` | resume | Skip already downloaded locations |
| `--workers` | `4` | Parallel worker threads (keep low for public Overpass API) |
| `--debug` | off | Enable DEBUG-level logging |

#### `create_graphs.py` — build graph pickles

```bash
# Default (CLIP embeddings)
python create_graphs.py \
  --output-dir location_dataset \
  --tagw-path data/all_tags30_frequency1.json

# SBERT backend — update node_embedding_dim in configs/default.yaml to match
python create_graphs.py --output-dir location_dataset \
  --tagw-path data/all_tags30_frequency1.json --embedding-backend sbert

# Force full rebuild
python create_graphs.py --output-dir location_dataset \
  --tagw-path data/all_tags30_frequency1.json --no-resume
```

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | required | Same directory used with `create_dataset.py` |
| `--tagw-path` | `data/all_tags30_frequency1.json` | Tag weights JSON for graph construction |
| `--embedding-backend` | `clip` | `clip` (512-dim) or `sbert` (384/768-dim) |
| `--embedding-model` | — | Override default model for the chosen backend |
| `--embedding-cache-path` | — | SQLite file for persistent embedding cache (keyed by model + word) |
| `--resume` / `--no-resume` | resume | Skip already built graph pickles |
| `--workers` | `4` | Parallel worker threads |
| `--debug` | off | Enable DEBUG-level logging |

#### Output files

| File | Written by | Description |
|---|---|---|
| `dataset.db` | both scripts | SQLite database with two tables: `downloads` (one row per downloaded location/level) and `graphs` (one row per built graph pickle) — consumed by training |
| `metadata.json` | both scripts | download-side fields written by `create_dataset.py`; graph-side fields (`tagw_path`, `embedding_backend`, `embedding_model`) merged in by `create_graphs.py` |

#### Adaptive bbox behaviour

When `--adaptive-bbox` is enabled the downloader expands the bbox until richness thresholds are met or the ceiling is reached. The actual bbox size used per sample is written to the `bbox_size` column in `dataset.db`.

Richness is assessed in `osmgraphclip/richness.py` on raw GeoDataFrames before graph construction. Metrics include tag entropy, category coverage, IDF-weighted score, geometry entropy, and spatial spread, combined into a single `richness_score` in [0, 1].

**Profile presets** (`--bbox-profile`) set both initial and maximum bbox size:

| Profile | Initial | Max | Typical use |
|---|---|---|---|
| `dense_city` | 100 m | 200 m | Dense urban |
| `suburb` | 250 m | 500 m | Suburban (default behaviour without profile) |
| `rural` | 500 m | 2 km | Villages, farmland |
| `wilderness` | 1 km | 5 km | Remote/sparse/global datasets |
| `regional` | 5 km | 20 km | Large rural or regional areas |

Explicit `--bbox-size` / `--max-bbox-size` override the profile. When no profile and no explicit values are given, the fallback is 250 m initial / 1000 m max.

**Profile escalation**: if a profile's ceiling is reached with *zero* data found, the next larger profile is tried automatically (continuing from the current bbox size). Escalation does not happen when some data exists but is below the richness thresholds — in that case the best available data is accepted. Log lines include the current profile label, e.g. `Location 0 [wilderness] bbox=1500m: nodes=4 ...`.

### Build a multi-scale band feature dataset

`create_multiscale_dataset.py` extends any location set with **concentric-ring (band) features** at multiple radii. For each location it:

1. Downloads fine-grain OSM data (`--bbox-size`, default 1000 m) and writes GeoJSON files compatible with `create_graphs.py`.
2. For each band radius downloads OSM data within a concentric disk, extracts spatial statistics and a distance-weighted SBERT embedding, then discards the intermediate GeoJSON.
3. Saves per-location band features as `<output-dir>/bands/osm_{i}_bands.npz`.

Run `create_graphs.py` afterwards to build graph pickles from the fine-grain GeoJSON.

```bash
# From CSV of locations (PostGIS backend recommended for large runs)
python create_multiscale_dataset.py \
  --output-dir multiscale_dataset \
  --locations-file data/locations.csv \
  --band-radii 2000 10000 20000 \
  --backend postgis \
  --postgis-url postgresql://user:pass@localhost/osm

# Built-in city grid
python create_multiscale_dataset.py \
  --output-dir multiscale_dataset \
  --city athens \
  --band-radii 2000 10000 20000

# Then build graph pickles
python create_graphs.py \
  --output-dir multiscale_dataset \
  --tagw-path data/all_tags30_frequency1.json \
  --embedding-backend sbert
```

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | required | Dataset output directory |
| `--locations-file` | — | CSV with `lat`/`lon` columns (mutually exclusive with `--city`) |
| `--city` | — | Built-in city grid |
| `--sample-spacing` | `500` | Grid spacing in metres (city mode) |
| `--bbox-size` | `1000` | Fine-grain bbox half-width in metres |
| `--band-radii` | `2000 10000 20000` | Band radii in metres (concentric disks) |
| `--embedding-model` | `all-MiniLM-L6-v2` | SBERT model for band embeddings |
| `--embedding-cache-path` | — | SQLite cache for SBERT embeddings |
| `--backend` | `overpass` | `overpass`, `postgis`, or `auto` |
| `--postgis-url` | — | PostgreSQL DSN (or `POSTGIS_URL` env var) |
| `--postgis-max-rows` | `50000` | Max rows per table for fine-grain queries |
| `--band-max-rows` | `100000` | SQL LIMIT per table for band queries (0 = unlimited) |
| `--device` | `cpu` | Torch device for SBERT model |
| `--resume` / `--no-resume` | resume | Skip already-processed locations |
| `--workers` | `4` | Parallel worker threads |
| `--debug` | off | Enable DEBUG-level logging |

#### Band .npz layout

Each `.npz` file at `<output-dir>/bands/osm_{i}_bands.npz` contains:

| Key | Shape | Description |
|---|---|---|
| `band_radii` | `[n_bands]` | Radii in metres |
| `spatial_features` | `[n_bands, 47]` | Global spatial statistics per band |
| `subbin_spatial` | `[n_bands, 2, 16]` | Inner / outer ring spatial features |
| `sector_spatial` | `[n_bands, 4, 11]` | N / E / S / W sector spatial features |
| `global_embeddings` | `[n_bands, D]` | Distance-weighted SBERT embedding |
| `subbin_embeddings` | `[n_bands, 2, D]` | Sub-bin SBERT embeddings |
| `sector_embeddings` | `[n_bands, 4, D]` | Sector SBERT embeddings |

### Generate globally-diverse training locations

`create_h3_locations.py` generates a globally-diverse CSV of `(lat, lon)` training locations using a four-phase pipeline backed by PostGIS:

1. **Density scan** — `TABLESAMPLE` query on `planet_osm_point` identifies non-empty 1-degree cells (~5–10 min, cached).
2. **H3 candidate generation** — enumerate all H3 res-5 cells (~22 km hexagons) whose centres fall within data-rich cells.
3. **Batch SQL scoring** — count OSM features per candidate using `VALUES + LATERAL` queries (100 locations per round-trip, no GeoDataFrame construction).
4. **Diversity-aware selection** — stratified sampling across density and spatial axes.

Output CSV is directly compatible with `create_dataset.py --locations-file`.

```bash
# Full run (PostGIS required)
python create_h3_locations.py \
  --output-dir h3_locations \
  --postgis-url postgresql://localhost/gis \
  --num-output-points 100000

# Quick smoke test
python create_h3_locations.py \
  --output-dir /tmp/h3_test \
  --postgis-url postgresql://localhost/gis \
  --num-output-points 500 \
  --debug

# Force re-run (ignore caches)
python create_h3_locations.py \
  --output-dir h3_locations \
  --postgis-url postgresql://localhost/gis \
  --no-resume
```

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | required | Output directory for CSV and caches |
| `--postgis-url` | — | PostgreSQL DSN (or `POSTGIS_URL` env var) |
| `--num-output-points` | `100000` | Number of locations to write to CSV |
| `--h3-resolution` | `5` | H3 cell resolution (5 = ~22 km hexagons) |
| `--jitter` | `0.05` | Random jitter in degrees added to cell centres |
| `--bbox-size` | `500` | Fixed bbox half-width for feature counting |
| `--min-total-features` | `5` | Min OSM features to include a candidate |
| `--batch-size` | `100` | Locations per SQL query round-trip |
| `--workers` | `4` | Parallel PostGIS connections |
| `--density-sample-pct` | `10.0` | TABLESAMPLE percentage for density scan |
| `--max-per-coarse-cell` | `10` | Max selected locations per H3 res-3 cell (~130 km) |
| `--seed` | `42` | Random seed |
| `--resume` / `--no-resume` | resume | Resume from existing caches |
| `--debug` | off | Enable DEBUG-level logging |

### Convert to LitData streaming format (optional)

`create_litdata_dataset.py` converts a finished graph-pickle dataset to LitData streaming chunks, producing `train/` and `val/` directories for use with `OsmLitDataModule`.

```bash
python create_litdata_dataset.py \
  --input-dir satclip_osm_dataset \
  --output-dir satclip_litdata \
  --workers 16

# Include band .npz files for multiscale training
python create_litdata_dataset.py \
  --input-dir multiscale_dataset \
  --output-dir multiscale_litdata \
  --include-bands \
  --workers 16
```

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | required | Source dataset directory (contains `dataset.db`) |
| `--output-dir` | required | LitData output root (`train/` and `val/` written here) |
| `--val-fraction` | `0.1` | Fraction of samples held out for validation |
| `--seed` | `17` | Random seed for train/val split |
| `--workers` | `8` | Parallel workers |
| `--chunk-size` | `128` | Graphs per LitData chunk file |
| `--resolution-mode` | `finest` | `all` (every level) or `finest` (one per location) |
| `--include-bands` | off | Bundle band `.npz` data alongside graph pickles (for `MultiscaleLitDataModule`) |

### Training

`train.py` uses LightningCLI — any config key can be overridden from the command line with dot notation.

```bash
# Standard graph + coordinate CLIP model
python train.py --config configs/default.yaml

# Multiscale model (graph + concentric-band features)
python train.py --config configs/multiscale.yaml

# Multiscale model with LitData streaming
python train.py --config configs/multiscale_litdata.yaml

# Override individual hyperparameters
python train.py --config configs/default.yaml --model.learning_rate 0.001
python train.py --config configs/default.yaml --data.batch_size 64

# Watch model graph in TensorBoard
python train.py --config configs/default.yaml --watchmodel
```

Training logs go to `logs/` (TensorBoard). Checkpoints saved per `val_loss`.

| Config | Model class | Data module | Use when |
|---|---|---|---|
| `configs/default.yaml` | `OSMGraphCLIPLightningModule` | `OsmGeoDataModule` | Standard graph-only training |
| `configs/multiscale.yaml` | `OSMGraphCLIPMultiscaleLightningModule` | `MultiscaleOsmGeoDataModule` | Graph + band features (pickle path) |
| `configs/multiscale_litdata.yaml` | `OSMGraphCLIPMultiscaleLightningModule` | `MultiscaleLitDataModule` | Graph + band features (LitData path) |

## Architecture

OSMGraphCLIP is a CLIP-style contrastive model that learns joint embeddings of OSM heterogeneous graphs and geographic coordinates.

### Data pipeline
1. `create_dataset.py` downloads OSM data (via Overpass or PostGIS) and writes typed GeoJSON files (`*_polygon.geojson.gz`, `*_linestring.geojson.gz`, `*_point.geojson.gz`) plus `dataset.db` (`downloads` table) and `metadata.json` into `--output-dir`. Downloader selection is handled by `osmgraphclip/downloader_config.py` with backends in `osmgraphclip/osm_downloader.py` (Overpass), `osmgraphclip/postgis_downloader.py` (PostGIS), and `osmgraphclip/auto_downloader.py` (auto-fallback).
2. `create_graphs.py` reads the `downloads` table from `dataset.db`, converts GeoJSON to `HeteroData` graph pickles using `OSM2Graph` (`osmgraphclip/osm_to_graph.py`), and populates the `graphs` table. Shared pipeline utilities live in `osmgraphclip/dataset_pipeline.py`. An optional SQLite embedding cache (`osmgraphclip/embedding_cache.py`) avoids redundant model inference across runs.
3. Pickles stored at `<output-dir>/graphs/osm_{i}_graph.pkl`; `dataset.db` and `metadata.json` sit in the dataset root.
4. `OsmDataset` / `OsmGeoDataModule` (`osmgraphclip/data/osm_dataset.py`) load pickles at training time. Batching uses `Batch.from_data_list` for heterogeneous graphs. An alternative high-throughput path uses `OsmLitDataModule` (`osmgraphclip/data/litdata_dataset.py`) with datasets pre-converted by `create_litdata_dataset.py`.

**Multiscale extension**: `create_multiscale_dataset.py` additionally writes per-location `.npz` band files under `<output-dir>/bands/`. `MultiscaleOsmGeoDataModule` (`osmgraphclip/data/multiscale_dataset.py`) and `MultiscaleLitDataModule` (`osmgraphclip/data/litdata_dataset.py`) pair graph pickles with band tensors at training time.

### Model components
- **`OSMGraphCLIP`** (`osmgraphclip/model.py`): top-level module combining graph and location encoders.
  - **`OSMHeteroGAT`** (`osmgraphclip/osm_encoder.py`): single-layer `HeteroConv` with `GATConv` over 3 node types (`point`, `line`, `polygon`), all 9 cross-type edge relations.
  - Graph outputs are pooled per type with `Set2Set`, projected with per-type MLPs, then attention-combined and projected to `embed_dim`.
  - **`LocationEncoder`** (`osmgraphclip/location_encoder.py`): positional encoding (default: spherical harmonics) → SIREN network → `embed_dim`.
  - Contrastive loss: symmetric cross-entropy on cosine similarity logits, temperature-scaled (`logit_scale`).
- **`OSMGraphCLIPLightningModule`** (`osmgraphclip/lightning_module.py`): Lightning wrapper; AdamW with weight decay excluded for biases, norms, and `logit_scale`.
- **`OSMGraphCLIPMultiscale`** (`osmgraphclip/model_multiscale.py`): extends `OSMGraphCLIP` with a `BandEncoder` transformer. When `band_data` is supplied the graph embedding and band embedding are concatenated and projected back to `embed_dim`; when `band_data` is `None` it falls back to plain `OSMGraphCLIP` behaviour.
  - **`BandEncoder`** (`osmgraphclip/band_encoder.py`): transformer encoder over multi-scale band tokens. Token layout: `[CLS]` + n_bands global tokens + n_bands×2 sub-bin tokens + n_bands×4 sector tokens. CLS output projected to `embed_dim`.
- **`OSMGraphCLIPMultiscaleLightningModule`** (`osmgraphclip/lightning_module_multiscale.py`): Lightning wrapper for the multiscale model; identical training loop to the standard module.

### Node feature dimensions (must match `node_embedding_dim` in config)
| Node type | Feature dim |
|-----------|------------|
| `point`   | `embedding_dim + 2` |
| `line`    | `embedding_dim + 6` |
| `polygon` | `embedding_dim + 8` |

### Embedding backends
| Backend | Default model | Dim | Config value |
|---------|--------------|-----|-------------|
| `clip` (default) | `openai/clip-vit-base-patch16` | 512 | `node_embedding_dim: 512` |
| `sbert` | `all-MiniLM-L6-v2` | 384 | `node_embedding_dim: 384` |
| `sbert` | `all-mpnet-base-v2` | 768 | `node_embedding_dim: 768` |

**`node_embedding_dim` in `configs/default.yaml` must match the backend used when building the dataset.**

### Inference / loading a checkpoint

**CLI script (`infer.py`)**:
```bash
# Print embedding to stdout (device auto-detected)
python infer.py --ckpt path/to/checkpoint.ckpt --lat 52.52 --lon 13.40

# Load directly from HuggingFace Hub
python infer.py --hf-model osmgraphclip-a-l10 --lat 52.52 --lon 13.40

# Lightweight loader (skips Lightning + graph encoder)
python infer.py --ckpt path/to/checkpoint.ckpt --lat 52.52 --lon 13.40 --lightweight

# HuggingFace + lightweight loader, JSON output
python infer.py --hf-model osmgraphclip-a-l10 --lat 52.52 --lon 13.40 \
    --lightweight --format json

# Output as JSON to stdout
python infer.py --ckpt path/to/checkpoint.ckpt --lat 52.52 --lon 13.40 --format json

# Save numpy array to file
python infer.py --ckpt path/to/checkpoint.ckpt --lat 52.52 --lon 13.40 \
    --format numpy --output embedding.npy

# Save JSON to file
python infer.py --ckpt path/to/checkpoint.ckpt --lat 52.52 --lon 13.40 \
    --format json --output embedding.json
```

`--ckpt` and `--hf-model` are mutually exclusive; exactly one is required.

| Flag | Default | Description |
|---|---|---|
| `--ckpt` | — | Path to a local Lightning checkpoint (mutually exclusive with `--hf-model`) |
| `--hf-model` | — | HuggingFace model name, e.g. `osmgraphclip-a-l10` (mutually exclusive with `--ckpt`) |
| `--lat` | required | Latitude in degrees |
| `--lon` | required | Longitude in degrees |
| `--lightweight` | off | Use lightweight loader (faster, no full Lightning load) |
| `--format` | `print` | Output format: `print`, `json`, or `numpy` |
| `--output` | stdout | File path to save embedding |
| `--device` | auto | Torch device override (`cpu`, `cuda`) |

**Available HuggingFace models** (hosted under `d-michail/`):

| Model name | Legendre polys | Multiscale |
|---|---|---|
| `osmgraphclip-a-l10` | 10 | no |
| `osmgraphclip-a-l40` | 40 | no |
| `osmgraphclip-ms-l10` | 10 | yes |
| `osmgraphclip-ms-l40` | 40 | yes |

**Python API**:
```python
from osmgraphclip.load import get_osmgraphclip

# Returns LocationEncoder only (default)
location_encoder = get_osmgraphclip("path/to/checkpoint.ckpt", device="cpu")

# Returns full OSMGraphCLIP
model = get_osmgraphclip("path/to/checkpoint.ckpt", device="cpu", return_all=True)

# Lightweight loader (skips graph encoder weights)
from osmgraphclip.load_lightweight import get_osmgraphclip_loc_encoder
location_encoder = get_osmgraphclip_loc_encoder("path/to/checkpoint.ckpt", device="cpu")
```

**HuggingFace Python API** — downloads checkpoint on first call, then uses the local HF cache:
```python
from osmgraphclip.load import get_osmgraphclip_from_hf

# Returns LocationEncoder only (default)
location_encoder = get_osmgraphclip_from_hf("osmgraphclip-a-l10", device="cpu")

# Returns full OSMGraphCLIP
model = get_osmgraphclip_from_hf("osmgraphclip-a-l10", device="cpu", return_all=True)

# Lightweight loader (skips graph encoder weights)
from osmgraphclip.load_lightweight import get_osmgraphclip_loc_encoder_from_hf
location_encoder = get_osmgraphclip_loc_encoder_from_hf("osmgraphclip-a-l10", device="cpu")
```

> **Note**: coords are passed as `(lon, lat)` order, shape `[N, 2]`, `float64`.

### Key config parameters (`configs/default.yaml`)
- `model.node_embedding_dim` — must match dataset embedding backend
- `model.le_type` — positional encoding type (`sphericalharmonics`, `grid`, `cartesian3d`, etc.)
- `model.pe_type` — neural network type (`siren`, `mlp`, `fcnet`)
- `model.legendre_polys` — resolution of spherical harmonics (10 = lower, 40 = higher)
- `data.data_dir` — path to dataset directory
- `data.mode` — `both` (graph + coords) or `points` (coords only)
- `data.resolution_mode` — `all` (use every level) or `finest` (one sample per location)
