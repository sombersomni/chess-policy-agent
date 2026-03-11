# chess-sim

**An encoder-decoder transformer for chess move prediction, trained via supervised learning on PGN master games and fine-tuned with offline reinforcement learning.**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-private-lightgrey)
![Ruff](https://img.shields.io/badge/linter-ruff-000000)

---

## Overview

chess-sim treats chess move generation as sequence-to-sequence translation. A 6-layer transformer encoder reads the board state as a 65-token sequence (CLS + 64 squares across three parallel embedding streams), producing a $65 \times d$ memory tensor. A 4-layer autoregressive decoder cross-attends to that memory and generates moves token-by-token from a 1971-token vocabulary.

Two training paths are supported:

| Path | Script | Loss | Data |
|------|--------|------|------|
| **Supervised Learning (SL)** | `scripts/train_v2.py` | Cross-entropy with teacher forcing | PGN / HDF5 |
| **Offline RL** | `scripts/train_rl.py` | RSBC + value loss | PGN master games |

**Current best checkpoint:** `chess_v2_1k.pt` -- $d_\text{model}=128$, 94.9% validation accuracy, 2.8M parameters.

---

## Architecture

### Encoder-Decoder Pipeline

```mermaid
graph LR
    PGN[".pgn / .pgn.zst"] --> TOK["Tokenizer<br/>board + color + trajectory"]
    TOK --> ENC["ChessEncoder<br/>6L, 8H, Pre-LN"]
    ENC -->|"memory [B,65,d]"| DEC["MoveDecoder<br/>4L, 8H, causal"]
    HIST["Move history<br/>SOS + prior moves"] --> DEC
    DEC -->|"logits [B,T,1971]"| PRED["argmax / sample"]
```

### Token Construction

Each board position is encoded as 65 tokens in three parallel streams:

| Stream | Vocab | Description |
|--------|-------|-------------|
| `board_tokens` | 8 | Piece type: 0=CLS, 1=empty, 2=pawn, 3=knight, 4=bishop, 5=rook, 6=queen, 7=king |
| `color_tokens` | 3 | Ownership: 0=empty/CLS, 1=player (side-to-move), 2=opponent |
| `trajectory_tokens` | 5 | Last-move roles: 0=none, 1=player prev src, 2=player prev tgt, 3=opp prev src, 4=opp prev tgt |

> **Note:** `square_emb` is an internal positional embedding (65 positions, sin/cos geometric init) -- not an external input stream.

### Embedding Layer

Four learned embeddings are summed element-wise and normalized:

$$\mathbf{h}_i = \text{LayerNorm}\!\Big(\mathbf{E}_{\text{piece}}[b_i] + \mathbf{E}_{\text{color}}[c_i] + \mathbf{E}_{\text{square}}[i] + \mathbf{E}_{\text{traj}}[t_i]\Big)$$

where $b_i$, $c_i$, $t_i$ are the piece, color, and trajectory token at position $i \in \{0, \ldots, 64\}$.

### Model Hyperparameters

| Component | `d_model` | Heads | Layers | FFN dim | Dropout |
|-----------|-----------|-------|--------|---------|---------|
| Encoder | 128 | 8 | 6 | 512 | 0.1 |
| Decoder | 128 | 8 | 4 | 512 | 0.1 |

> **Note:** The default `ModelConfig` specifies $d_\text{model}=256$, but the current best checkpoint uses $d_\text{model}=128$ (3.6x fewer parameters with only 0.7% accuracy loss vs. 256).

**Move vocabulary:** 1971 tokens covering all legal UCI move strings (including promotions).

---

## Training: Supervised Learning

The SL path trains the decoder with teacher forcing, minimizing cross-entropy between predicted and actual moves:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta\!\big(m_t \mid m_{<t},\, \mathbf{s}\big)$$

where $m_t$ is the ground-truth move at ply $t$, $m_{<t}$ is the move prefix, and $\mathbf{s}$ is the encoder memory from the board state.

Optional label smoothing ($\epsilon$) redistributes probability mass to non-target tokens, reducing overfitting on small datasets.

```bash
source .venv/bin/activate

# Train from HDF5 (recommended -- preprocessed data, fast I/O)
python -m scripts.train_v2 --config configs/train_v2_10k.yaml --hdf5 data/processed/chess_dataset.h5

# Train from raw PGN (on-the-fly tokenization)
python -m scripts.train_real --pgn data/games.pgn --epochs 10 --checkpoint checkpoints/run_01.pt
```

### Checkpoint Results

| Checkpoint | Data | $d_\text{model}$ | Epochs | Val Loss | Val Acc |  Params |
|---|---|---|---|---|---|---|
| `chess_v2_1k.pt` | 1k games | 128 | 20 | 0.229 | **94.9%** |  2.8M |
| `chess_v2_10k.pt` | 10k games | 128 | 20 | 2.606* | 62.4% |  2.8M |

*\*Val loss inflated by `label_smoothing=0.1` -- not directly comparable to unsmoothed runs.*

---

## Training: Offline RL

The offline RL path fine-tunes a pretrained checkpoint on PGN master games using **Result-Scaled Behavioral Cloning (RSBC)** combined with a learned **action-conditioned value head**.

### Reward Function

Each ply $t$ receives a composite reward:

$$R(t) = \lambda_{\text{outcome}} \cdot \text{sign}(t) + \lambda_{\text{material}} \cdot \Delta_{\text{material}}(t)$$

where:

| Symbol | Definition | Default |
|--------|-----------|---------|
| $\text{sign}(t)$ | $+1$ (winner ply), $-1$ (loser ply), or $d$ (draw ply) | -- |
| $d$ | `draw_reward_norm` | 0.5 |
| $\lambda_{\text{outcome}}$ | Outcome signal weight | 1.0 |
| $\lambda_{\text{material}}$ | Material delta weight | 0.1 |
| $\Delta_{\text{material}}(t)$ | Piece-value change at ply $t$ (player perspective) | -- |

### Loss Function

The total training loss combines outcome-weighted behavioral cloning with a value regression term:

$$\mathcal{L} = \lambda_{\text{rsbc}} \cdot \mathcal{L}_{\text{RSBC}} + \lambda_{\text{value}} \cdot \mathcal{L}_{\text{value}}$$

**RSBC loss** applies per-ply CE weighted by game outcome:

$$\mathcal{L}_{\text{RSBC}} = \frac{1}{N}\sum_{i=1}^{N} w_i \cdot \text{CE}\!\big(\hat{\mathbf{y}}_i,\, m_i\big)$$

| Ply type | Weight $w_i$ |
|----------|-------------|
| Winner | 1.0 |
| Draw | `draw_reward_norm` (0.5) |
| Loser | `loser_ply_weight` (0.1) |

**Value loss** trains a Q-head to predict discounted returns:

$$\mathcal{L}_{\text{value}} = \text{MSE}\!\big(Q_\phi(\mathbf{z}_{\text{cls}}, \mathbf{e}_a),\; R(t)\big)$$

where $\mathbf{z}_{\text{cls}}$ is the encoder CLS embedding and $\mathbf{e}_a$ is the action (move token) embedding.

### Running RL Training

```bash
source .venv/bin/activate
python -m scripts.train_rl --config configs/train_rl.yaml
```

### RL Configuration (`configs/train_rl.yaml`)

```yaml
data:
  pgn:       data/lichess_db_standard_rated_2013-01.pgn.zst
  max_games: 1000

model:
  d_model: 128
  n_heads: 8
  n_layers: 6
  dim_feedforward: 512

decoder:
  d_model: 128
  n_heads: 8
  n_layers: 4
  dim_feedforward: 512
  move_vocab_size: 1971

rl:
  learning_rate:    0.0001
  epochs:           20
  lambda_rsbc:      1.0
  lambda_value:     1.0
  lambda_outcome:   1.0
  lambda_material:  0.1
  draw_reward_norm: 0.5
  loser_ply_weight: 0.1
  balance_outcomes: true
  train_color:      white
  checkpoint:       checkpoints/chess_rl.pt
```

### LR Schedule

The RL trainer uses a three-phase schedule:

$$\text{lr}(t) = \begin{cases}
\text{lr}_{\max} \cdot \frac{t}{t_{\text{warmup}}} & t < t_{\text{warmup}} \\[4pt]
\text{lr}_{\max} & t_{\text{warmup}} \le t < t_{\text{decay}} \\[4pt]
\text{lr}_{\min} + \tfrac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})\!\left(1 + \cos\!\left(\pi \cdot \frac{t - t_{\text{decay}}}{t_{\text{total}} - t_{\text{decay}}}\right)\right) & t \ge t_{\text{decay}}
\end{cases}$$

where $t_{\text{warmup}} = \lfloor \texttt{warmup\_fraction} \cdot t_{\text{total}} \rfloor$ and $t_{\text{decay}} = \lfloor \texttt{decay\_start\_fraction} \cdot t_{\text{total}} \rfloor$.

The value head receives a separate learning rate: $\text{lr}_{\text{value}} = \text{lr}_{\max} \times \texttt{value\_lr\_multiplier}$ (default 5x).

---

## Evaluation

```bash
python -m scripts.evaluate \
    --checkpoint checkpoints/chess_v2_1k.pt \
    --pgn data/games.pgn \
    --game-index 0 \
    --top-n 3
```

Output includes a per-ply table with CE loss, top-1 accuracy, and Shannon entropy $H$ for each prediction head:

$$H = -\sum_{i=1}^{V} p_i \log p_i$$

where $V = 1971$ is the move vocabulary size. Higher entropy indicates the model is less certain about the move.

Use `--winners-only` to restrict evaluation to positions where the winning player is to move.

---

## Data Pipeline

```mermaid
graph LR
    PGN[".pgn / .pgn.zst"] --> STREAM["StreamingPGNReader"]
    STREAM --> SAMPLE["ReservoirSampler<br/>(Vitter's Algorithm R)"]
    SAMPLE --> TOK["BoardTokenizer"]
    TOK --> DS["ChessDataset / HDF5"]
    DS --> DL["DataLoader"]
```

### 1. Stream games

```python
from chess_sim.data.reader import StreamingPGNReader
reader = StreamingPGNReader()
for game in reader.stream(Path("lichess_db.pgn.zst")):
    process(game)
```

### 2. Sample uniformly at random

```python
from chess_sim.data.sampler import ReservoirSampler
sampler = ReservoirSampler()
games = sampler.sample(reader.stream(path), n=1_000_000)
```

### 3. Tokenize a board position

```python
import chess
from chess_sim.data.tokenizer import BoardTokenizer

tok = BoardTokenizer()
board = chess.Board()
result = tok.tokenize(board, chess.WHITE)
# result.board_tokens  -> list[int], length 65
# result.color_tokens  -> list[int], length 65
```

### 4. Generate trajectory tokens

```python
from chess_sim.data.tokenizer_utils import make_trajectory_tokens
trajectory_tokens = make_trajectory_tokens(move_history)
# list[int], length 65; values in {0,1,2,3,4}
```

### 5. Build a DataLoader

```python
from torch.utils.data import DataLoader
from chess_sim.data.dataset import ChessDataset

train_ds, val_ds = ChessDataset.split(examples, train_frac=0.95)
loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
```

---

## Data Preparation (HDF5 Preprocessing)

Training from raw PGN is slow (re-parses every run). The preprocessor writes all tokenized records to HDF5 once; subsequent runs read pre-baked integer arrays.

```bash
source .venv/bin/activate

# Full dataset
python -m scripts.preprocess --config configs/preprocess_v2.yaml

# Smoke test (100 games)
python -m scripts.preprocess \
    --config configs/preprocess_v2.yaml \
    --max-games 100 \
    --output data/processed/chess_dataset_small.h5
```

### HDF5 Schema

Each row is one board state at a specific game ply.

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `board_tokens` | `[N, 65]` | `uint8` | Piece type per square (CLS at index 0) |
| `color_tokens` | `[N, 65]` | `uint8` | Piece ownership per square |
| `trajectory_tokens` | `[N, 65]` | `uint8` | Last-2-move trajectory roles |
| `move_tokens` | `[N, 512]` | `uint16` | Decoder input: SOS + prior moves (padded) |
| `target_tokens` | `[N, 512]` | `uint16` | Decoder targets (padded) |
| `move_lengths` | `[N]` | `uint16` | Actual sequence length before padding |
| `outcome` | `[N]` | `int8` | +1 win / 0 draw / -1 loss (player-to-move) |
| `turn` | `[N]` | `uint16` | 0-indexed ply within the game |
| `game_id` | `[N]` | `uint32` | Parent game index |

Split groups: `train/` and `val/` (default 95/5 split, deterministic by `game_id`).

---

## Terminal Simulation

Replay games, generate random playthroughs, or watch the model predict moves in real time.

```
8 r n b q k b n r   Ply 2  -  e7e5
7 p p p p . p p p   Phase: opening
6 . . . . . . . .   Material: 0
5 . . . . p . . .
4 . . . . P . . .   Move history:
3 . . . . . . . .     1. e2e4
2 P P P P . P P P     2. e7e5
1 R N B Q K B N R
  a b c d e f g h   -- Agent Predictions --
                      1. e7e5    42.1%  correct
                      2. c7c5    18.3%
                      3. g8f6     9.7%
```

| Mode | Command | Description |
|------|---------|-------------|
| `pgn` | `--mode pgn --pgn data/games.pgn.zst` | Replay a recorded game |
| `random` | `--mode random` | Random legal moves |
| `agent` | `--mode agent --pgn ... --checkpoint ...` | Model predictions before each move |

```bash
source .venv/bin/activate

# PGN replay
python -m scripts.simulate --mode pgn \
    --pgn data/lichess_db_standard_rated_2013-01.pgn.zst \
    --game-index 0 --tick-rate 0.5

# Agent prediction mode
python -m scripts.simulate --mode agent \
    --pgn data/lichess_db_standard_rated_2013-01.pgn.zst \
    --checkpoint checkpoints/chess_v2_1k.pt \
    --tick-rate 1.0 --top-n 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | -- | Path to `configs/simulate.yaml` |
| `--mode` | -- | `pgn` / `random` / `agent` (required) |
| `--pgn` | -- | Path to `.pgn` or `.pgn.zst` file |
| `--game-index` | `0` | Zero-based game index in the PGN |
| `--checkpoint` | -- | Path to `.pt` checkpoint (agent mode) |
| `--tick-rate` | `0.5` | Seconds between plies |
| `--top-n` | `3` | Number of predictions to display |
| `--max-plies` | `200` | Truncation limit (random mode) |
| `--no-unicode` | -- | Use ASCII piece symbols |

The environment satisfies `gymnasium.Env` (`obs=(65,3) float32`, `action=Discrete(1971)`).

---

## GUI Viewer

Step through a game visually: animated board on the left, per-ply metrics (loss, accuracy, entropy) and PCA embedding scatter on the right.

```bash
source .venv/bin/activate
python -m scripts.gui.viewer \
    --pgn data/games.pgn \
    --checkpoint checkpoints/chess_v2_1k.pt \
    --game-index 0
```

Controls: `Prev` / `Next` buttons or drag the slider.

---

## Setup

### Prerequisites

- Python 3.10+
- `virtualenv`

### Install

```bash
cd chess-sim
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Linting

Uses [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`).

| Code | Category | What it catches |
|------|----------|----------------|
| `E` | pycodestyle errors | Line length (>88 chars), whitespace, indentation |
| `W` | pycodestyle warnings | Trailing whitespace |
| `F` | pyflakes | Unused imports, undefined names |
| `ANN` | flake8-annotations | Missing type annotations (`tests/` exempt) |
| `I` | isort | Unsorted imports |

```bash
source .venv/bin/activate
python -m ruff check .           # Check
python -m ruff check . --fix     # Auto-fix safe issues
python -m ruff check . --statistics  # Summary by rule
```

---

## Tests

All tests are CPU-only and deterministic.

```bash
source .venv/bin/activate
python -m unittest discover -s tests -p "test_*.py"
```

| File | Coverage |
|------|----------|
| `tests/test_tokenizer.py` | T01--T04: BoardTokenizer correctness |
| `tests/test_embedding.py` | T05--T08: EmbeddingLayer shapes and dtype |
| `tests/test_encoder.py` | T09--T12: ChessEncoder forward pass and gradients |
| `tests/test_heads.py` | T13--T14: PredictionHeads output shapes |
| `tests/test_loss.py` | T15--T16: LossComputer correctness |
| `tests/test_trainer.py` | T19: train_step, checkpoint roundtrip |
| `tests/test_dataset.py` | T17, T20: DataLoader dtypes and move labels |
| `tests/test_reader.py` | StreamingPGNReader streaming |
| `tests/test_sampler.py` | ReservoirSampler uniform sampling |
| `tests/test_chess_encoder.py` | T26--T40: trajectory tokens, embedding init, gradient flow |
| `tests/test_evaluate.py` | TEV01--TEV14: entropy, accuracy, per-head CE, GameEvaluator |
| `tests/env/test_chess_sim_env.py` | T1--T12: PGNSource, RandomSource, ChessSimEnv, gymnasium env_checker |

---

## CI Pipeline

Every PR triggers a Jenkins build on the Kubernetes cluster (`10.0.0.169`).

```mermaid
graph LR
    PR["GitHub PR"] -->|webhook| JK["Jenkins<br/>K8s cluster"]
    JK --> POD["chess-sim:ci pod"]
    POD --> LINT["ruff check"]
    LINT --> TEST["unittest"]
    TEST -->|status| GH["GitHub PR"]
```

### Accessing Jenkins

| URL | Port | Notes |
|-----|------|-------|
| `https://jenkins.local:30443` | 30443 | Primary HTTPS (requires `/etc/hosts` entry) |
| `https://10.0.0.169:30443` | 30443 | Direct IP access |
| `http://10.0.0.169:30080` | 30080 | Legacy HTTP NodePort |

Add to `/etc/hosts`:
```
10.0.0.169  jenkins.local
```

Trust the home-lab CA:
```bash
kubectl get secret jenkins-ca-secret -n cert-manager \
  -o jsonpath='{.data.tls\.crt}' | base64 -d > jenkins-ca.crt
# Import jenkins-ca.crt into your OS / browser CA trust store.
```

### Infrastructure

All Jenkins infrastructure is Helm-managed under `jenkins/`.

| File | Purpose |
|------|---------|
| `jenkins/values.yaml` | Jenkins Helm values |
| `jenkins/ingress-controller-values.yaml` | nginx ingress (HTTPS on 30443) |
| `jenkins/cert-manager-values.yaml` | cert-manager with CRDs |
| `jenkins/tls.yaml` | Self-signed CA, leaf cert, Ingress |
| `jenkins/pod-template.yaml` | K8s agent pod spec |
| `jenkins/job-config.xml` | Jenkins job XML |

<details>
<summary>Install or reinstall the full stack</summary>

```bash
HELM=/home/sombersomni/bin/helm

# 1. Jenkins
$HELM repo add jenkins https://charts.jenkins.io
$HELM upgrade --install jenkins jenkins/jenkins \
    --namespace jenkins --create-namespace \
    --values jenkins/values.yaml

# 2. nginx ingress controller
$HELM repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
$HELM upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
    --namespace ingress-nginx --create-namespace \
    --values jenkins/ingress-controller-values.yaml

# 3. cert-manager
$HELM repo add jetstack https://charts.jetstack.io
$HELM upgrade --install cert-manager jetstack/cert-manager \
    --namespace cert-manager --create-namespace \
    --values jenkins/cert-manager-values.yaml

# 4. TLS resources
kubectl wait --for=condition=ready pod \
    -l app.kubernetes.io/instance=cert-manager \
    -n cert-manager --timeout=120s
kubectl apply -f jenkins/tls.yaml
```

</details>

### CI Image Versioning

The CI image (`Dockerfile.ci`) contains only Python deps. Rebuild when `requirements.txt`, base image, or system packages change.

| Tag | Example | Use |
|---|---|---|
| `ci-vMAJOR.MINOR.PATCH` | `ci-v1.0.0` | Pinned release (use in pod-template) |
| `ci-sha-<7>` | `ci-sha-a3f9c1` | Immutable per-commit tag |
| `ci-latest` | `ci-latest` | Convenience alias (never pin to this) |

```bash
VERSION=ci-v1.0.0
SHA=ci-sha-$(git rev-parse --short HEAD)
REPO=ghcr.io/$(gh api user --jq .login)/chess-sim

gh auth token | docker login ghcr.io -u $(gh api user --jq .login) --password-stdin

docker build -f Dockerfile.ci \
    -t ${REPO}:${VERSION} \
    -t ${REPO}:${SHA} \
    -t ${REPO}:ci-latest .

docker push ${REPO}:${VERSION}
docker push ${REPO}:${SHA}
docker push ${REPO}:ci-latest
```

After pushing, update `jenkins/pod-template.yaml` to the new `ci-vX.Y.Z` tag.

---

## Project Structure

```
chess-sim/
├── chess_sim/
│   ├── config.py              # Typed dataclass configs (YAML-loadable)
│   ├── protocols.py           # Structural type protocols
│   ├── types.py               # NamedTuple containers
│   ├── utils.py               # winner_color() helper
│   ├── data/
│   │   ├── tokenizer.py       # BoardTokenizer: Board -> TokenizedBoard
│   │   ├── tokenizer_utils.py # make_trajectory_tokens()
│   │   ├── reader.py          # StreamingPGNReader: .zst -> Game iterator
│   │   ├── sampler.py         # ReservoirSampler (Vitter's Algorithm R)
│   │   ├── dataset.py         # ChessDataset + train/val split
│   │   ├── pgn_sequence_dataset.py  # V2 on-the-fly PGN dataset
│   │   ├── hdf5_dataset.py    # ChessHDF5Dataset (pre-baked HDF5)
│   │   ├── move_tokenizer.py  # UCI string <-> vocab index
│   │   └── move_vocab.py      # 1971-token move vocabulary
│   ├── model/
│   │   ├── embedding.py       # EmbeddingLayer (piece+color+square+traj)
│   │   ├── encoder.py         # ChessEncoder (6-layer transformer)
│   │   ├── decoder.py         # MoveDecoder (4-layer, causal)
│   │   ├── chess_model.py     # ChessModel (top-level encoder-decoder)
│   │   ├── heads.py           # PredictionHeads (src/tgt square)
│   │   └── value_heads.py     # ActionConditionedValueHead (RL)
│   ├── env/
│   │   ├── sources.py         # PGNSource, RandomSource
│   │   ├── chess_sim_env.py   # ChessSimEnv(gym.Env)
│   │   ├── terminal_renderer.py  # Unicode board + ANSI highlights
│   │   └── agent_adapter.py   # ChessModelAgent (Policy protocol)
│   └── training/
│       ├── trainer.py         # SL Trainer (AdamW + cosine LR)
│       ├── loss.py            # LossComputer (CE x2)
│       ├── pgn_rl_trainer.py  # PGNRLTrainer (offline RL)
│       ├── pgn_replayer.py    # PGN game -> OfflinePlyTuple list
│       ├── pgn_rl_reward_computer.py  # Composite reward R(t)
│       └── training_utils.py  # Shared training utilities
├── scripts/
│   ├── preprocess.py          # PGN -> HDF5 (run once)
│   ├── train_v2.py            # V2 SL training
│   ├── train_real.py          # V1 SL training
│   ├── train_rl.py            # Offline RL training
│   ├── evaluate.py            # Per-move evaluation
│   ├── simulate.py            # Terminal simulation
│   └── gui/                   # Tkinter game viewer
├── configs/                   # YAML configuration files
├── checkpoints/               # Trained .pt files (gitignored)
├── data/                      # PGN files (gitignored)
├── tests/                     # Unit tests (T01-T40, TEV01-TEV14)
├── requirements.txt
└── chess_encoder_final_design.md
```

---

## Key Design Decisions

**Player-perspective prediction.** Each ply is predicted from the side-to-move's perspective. When it is Black's turn, the board is tokenized with Black as "player" and White as "opponent." The model always sees itself as the player.

**Pre-Layer-Norm (Pre-LN).** `TransformerEncoderLayer` uses `norm_first=True` for stable gradient flow with PyTorch 2.x's scaled dot-product attention.

**Square indexing.** Fixed geometric order: a1=1, b1=2, ..., h8=64. The board is never flipped; `color_tokens` convey piece ownership relative to the side-to-move.

**Trajectory tokens.** The trajectory stream encodes the role of each square in the last two half-moves. Opponent marks overwrite player marks on collision (correct for captures).

**`--winners-only` / `--winners-side` flags.** Training and evaluation can filter to positions where the game's winner is to move. `winners_side=true` keeps all plies from draws (both players are non-losers) while skipping loser plies from decisive games.

**Security.** All `torch.load()` calls use `weights_only=True` to prevent pickle-based arbitrary code execution. All YAML loading uses `yaml.safe_load()`.
