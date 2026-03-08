# Chess Encoder-Decoder — PyTorch Implementation Design

## 1. Module Overview

```
chess/
├── configs/
│   ├── base.yaml         # Base model + training config (all defaults)
│   ├── small.yaml        # Small model for fast experimentation
│   └── phase2.yaml       # RL fine-tuning config (inherits base.yaml)
├── vocab.py              # Vocabulary definitions and mappings
├── tokenizer.py          # Board and move tokenization
├── embeddings.py         # Piece embeddings, 2D positional encoding, move embeddings
├── encoder.py            # Board encoder (Transformer encoder stack)
├── decoder.py            # Move decoder (Transformer decoder stack)
├── model.py              # ChessModel — top-level encoder-decoder assembly
├── config.py             # YAML loading, validation, and config dataclass
├── dataset.py            # PGN dataset, collation, batching
└── train.py              # Training loop — reads config, runs to completion
```

---

## 2. Vocabulary — `vocab.py`

### Piece Vocabulary

```python
PIECE_TOKENS = {
    'P': 0,   # White Pawn
    'N': 1,   # White Knight
    'B': 2,   # White Bishop
    'R': 3,   # White Rook
    'Q': 4,   # White Queen
    'K': 5,   # White King
    'p': 6,   # Black Pawn
    'n': 7,   # Black Knight
    'b': 8,   # Black Bishop
    'r': 9,   # Black Rook
    'q': 10,  # Black Queen
    'k': 11,  # Black King
    'EMPTY': 12,
    'CLS':   13,
}

PIECE_VOCAB_SIZE = 14   # 13 piece types + CLS
EMPTY_IDX       = 12
CLS_IDX         = 13
```

### Move Vocabulary

```python
MOVE_SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
}

# All ~1968 legal UCI moves are enumerated and assigned indices 3..N
# e.g. { 'e2e4': 3, 'e7e5': 4, 'g1f3': 5, ... }
# Promotions included: e7e8q, e7e8r, e7e8b, e7e8n

MOVE_VOCAB_SIZE = len(MOVE_SPECIAL_TOKENS) + 1968   # ~1971
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
```

---

## 3. Tokenizer — `tokenizer.py`

### Board Tokenization

Converts a board state $B_t$ into a tensor of 64 piece token indices, ordered a1→h8.

```python
def tokenize_board(board: chess.Board) -> torch.Tensor:
    """
    Args:
        board: python-chess Board object

    Returns:
        tokens: LongTensor [64]
                index 0 = a1, index 63 = h8
                values in PIECE_TOKENS
    """
```

**Square ordering:**
```
index = rank * 8 + file
rank ∈ {0,...,7}  (0 = rank 1, 7 = rank 8)
file ∈ {0,...,7}  (0 = file a, 7 = file h)
```

### Move Tokenization

```python
def tokenize_move(move: str) -> int:
    """UCI string -> move vocabulary index"""

def tokenize_game(pgn_game) -> torch.Tensor:
    """
    Returns:
        moves: LongTensor [T+2]   (<SOS>, m_1, ..., m_T, <EOS>)
    """
```

---

## 4. Embeddings — `embeddings.py`

### 4.1 Board Input Embedding

Combines piece token embedding with factored 2D positional encoding.

```python
class BoardEmbedding(nn.Module):
    def __init__(self, d_model: int):
        self.piece_embedding = nn.Embedding(PIECE_VOCAB_SIZE, d_model)
        self.rank_embedding  = nn.Embedding(8, d_model)   # ranks 0-7
        self.file_embedding  = nn.Embedding(8, d_model)   # files 0-7
        self.cls_token       = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, board_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            board_tokens: LongTensor [B, 64]

        Returns:
            x: FloatTensor [B, 65, d_model]
               index 0   = CLS token
               index 1..64 = square embeddings
        """
        B = board_tokens.size(0)

        # Square embeddings
        piece_emb = self.piece_embedding(board_tokens)         # [B, 64, d_model]
        ranks = torch.arange(64) // 8                         # [64]
        files = torch.arange(64) % 8                          # [64]
        pos_emb = self.rank_embedding(ranks) \
                + self.file_embedding(files)                   # [64, d_model]
        square_emb = piece_emb + pos_emb.unsqueeze(0)         # [B, 64, d_model]

        # Prepend CLS
        cls = self.cls_token.expand(B, 1, d_model)            # [B, 1, d_model]
        x = torch.cat([cls, square_emb], dim=1)               # [B, 65, d_model]
        return x
```

**Key design note:** rank and file embeddings are added independently, not concatenated. This preserves the factored 2D structure — the model can learn rank-specific and file-specific patterns separately.

### 4.2 Move Sequence Embedding

```python
class MoveEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512):
        self.move_embedding = nn.Embedding(MOVE_VOCAB_SIZE, d_model,
                                           padding_idx=PAD_IDX)
        self.pos_embedding  = nn.Embedding(max_seq_len, d_model)

    def forward(self, move_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            move_tokens: LongTensor [B, T]

        Returns:
            x: FloatTensor [B, T, d_model]
        """
        positions = torch.arange(move_tokens.size(1))          # [T]
        x = self.move_embedding(move_tokens) \
          + self.pos_embedding(positions)                       # [B, T, d_model]
        return x
```

---

## 5. Board Encoder — `encoder.py`

Standard Transformer encoder. Stateless — takes only the current board.

```python
class BoardEncoder(nn.Module):
    def __init__(
        self,
        d_model:     int = 128,
        n_heads:     int = 8,
        n_layers:    int = 4,
        d_ff:        int = 512,
        dropout:     float = 0.1,
    ):
        self.embedding = BoardEmbedding(d_model)
        self.layers    = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_ff, dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        board_tokens: torch.Tensor             # [B, 64]
    ) -> torch.Tensor:
        """
        Returns:
            H: FloatTensor [B, 65, d_model]
               index 0     = CLS token
               index 1..64 = all squares including EMPTY
        """
        x = self.embedding(board_tokens)       # [B, 65, d_model]
        H = self.layers(x)                     # [B, 65, d_model]
        H = self.norm(H)
        return H
```

**Shape flow:**
```
board_tokens     [B, 64]
→ embedding      [B, 65, d_model]    (CLS prepended)
→ encoder layers [B, 65, d_model]    (full self-attention over all 65 tokens)
→ output         [B, 65, d_model]    (fixed shape, EMPTY tokens retained)
```

---

## 6. Move Decoder — `decoder.py`

Transformer decoder. Stateful — takes the full move history and cross-attends to encoder output.

```python
class MoveDecoder(nn.Module):
    def __init__(
        self,
        d_model:  int   = 128,
        n_heads:  int   = 8,
        n_layers: int   = 4,
        d_ff:     int   = 512,
        dropout:  float = 0.1,
    ):
        self.embedding = MoveEmbedding(d_model)
        self.layers    = nn.TransformerDecoder(
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_ff, dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.norm    = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, MOVE_VOCAB_SIZE)

    def forward(
        self,
        move_tokens:      torch.Tensor,   # [B, T]
        H_filtered:       torch.Tensor,   # [B, S, d_model]
        tgt_key_padding_mask:  torch.Tensor = None,   # [B, T]
    ) -> torch.Tensor:
        """
        Returns:
            logits: FloatTensor [B, T, MOVE_VOCAB_SIZE]
        """
        tgt = self.embedding(move_tokens)                  # [B, T, d_model]
        causal_mask = self._causal_mask(T)                 # [T, T]

        out = self.layers(
            tgt,
            memory=H_filtered,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )                                                  # [B, T, d_model]

        out    = self.norm(out)
        logits = self.out_proj(out)                        # [B, T, MOVE_VOCAB_SIZE]
        return logits

    def _causal_mask(self, T: int) -> torch.Tensor:
        """
        Upper triangular mask — prevents attending to future move tokens.
        Returns BoolTensor [T, T], True = masked (ignored)
        """
        return torch.triu(torch.ones(T, T), diagonal=1).bool()
```

**Shape flow:**
```
move_tokens      [B, T]
→ embedding      [B, T, d_model]
→ decoder layers [B, T, d_model]    causal self-attn + cross-attn to H_filtered
→ out_proj       [B, T, MOVE_VOCAB_SIZE]
```

**Three attention operations per decoder layer:**
```
1. Causal self-attention    Q=K=V=move_emb       [B, T, d_model]
2. Cross-attention          Q=move_emb           [B, T, d_model]
                            K=V=H_filtered        [B, S, d_model]
3. Feedforward              position-wise MLP
```

---

## 7. Top-Level Model — `model.py`

```python
class ChessModel(nn.Module):
    def __init__(self, config: ChessModelConfig):
        self.encoder = BoardEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.encoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        self.decoder = MoveDecoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.decoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

    def forward(
        self,
        board_tokens: torch.Tensor,   # [B, 64]
        move_tokens:  torch.Tensor,   # [B, T]
        move_pad_mask: torch.Tensor = None,  # [B, T]
    ) -> torch.Tensor:
        """
        Full forward pass.

        Returns:
            logits: FloatTensor [B, T, MOVE_VOCAB_SIZE]
        """
        H = self.encoder(board_tokens)        # [B, 65, d_model]
        logits = self.decoder(
            move_tokens,
            H,
            tgt_key_padding_mask=move_pad_mask,
        )
        return logits

    @torch.no_grad()
    def predict_next_move(
        self,
        board:       chess.Board,
        move_history: List[str],
        legal_moves:  List[str],
        temperature: float = 1.0,
    ) -> str:
        """
        Inference — predicts a single next move.
        Legal move masking applied before sampling.

        Returns:
            move: UCI string
        """
        board_tokens = tokenize_board(board).unsqueeze(0)      # [1, 64]
        move_tokens  = tokenize_game_prefix(move_history)       # [1, T]

        logits = self.forward(board_tokens, move_tokens)        # [1, T, V]
        next_logits = logits[0, -1, :]                          # [V]

        # Mask illegal moves
        legal_mask = build_legal_mask(legal_moves)              # [V] bool
        next_logits[~legal_mask] = -1e9

        probs = F.softmax(next_logits / temperature, dim=-1)
        move_idx = torch.multinomial(probs, num_samples=1).item()
        return MOVE_VOCAB_INV[move_idx]
```

---

## 8. Dataset — `dataset.py`

```python
class PGNDataset(Dataset):
    """
    Each sample is a single game turn:
        board_tokens:  LongTensor [64]       board state before move t
        move_tokens:   LongTensor [T]        [<SOS>, m_1, ..., m_{t-1}]
        target:        LongTensor [T]        [m_1, ..., m_t]  (shifted by 1)
        outcome:       float                 +1 win, 0 draw, -1 loss  (for Phase 2)
    """

def collate_fn(batch):
    """
    Pads move sequences to the longest in the batch.
    Returns:
        board_tokens:  [B, 64]
        move_tokens:   [B, T_max]
        targets:       [B, T_max]
        move_pad_mask: [B, T_max]  True = PAD position
        outcomes:      [B]
    """
```

---

## 9. Training — `train.py`

### Phase 1 — Supervised Pretraining

```python
def train_step_phase1(
    model:     ChessModel,
    batch:     Batch,
    optimizer: torch.optim.Optimizer,
) -> float:

    logits = model(
        board_tokens=batch.board_tokens,     # [B, 64]
        move_tokens=batch.move_tokens,       # [B, T]
        move_pad_mask=batch.move_pad_mask,   # [B, T]
    )                                        # [B, T, V]

    # Flatten for cross entropy
    logits  = logits.view(-1, MOVE_VOCAB_SIZE)   # [B*T, V]
    targets = batch.targets.view(-1)              # [B*T]

    loss = F.cross_entropy(logits, targets, ignore_index=PAD_IDX)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
```

### Phase 2 — REINFORCE Self-Play

```python
def train_step_phase2(
    model:    ChessModel,
    game:     SelfPlayGame,            # completed self-play game
    optimizer: torch.optim.Optimizer,
    delta:    float = 0.05,            # novelty bonus weight
) -> float:
    """
    REINFORCE update:
        loss = -sum_t [ R_t * log P(m_t | m_1,...,m_{t-1}, B_{t-1}) ]

    R_t is the per-move reward:
        outcome reward:  +1 win / -1 loss / 0 draw
        novelty bonus:   +delta if move not in top-k human frequency
    """
    rewards  = compute_rewards(game, delta)       # [T]
    baseline = rewards.mean()                     # variance reduction
    rewards  = rewards - baseline

    log_probs = compute_log_probs(model, game)    # [T]
    loss = -(rewards * log_probs).sum()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
```

---

## 10. Config — `configs/` + `config.py`

All training runs are defined by a YAML file. A developer should be able to clone the repo, write a config, and run:

```bash
python train.py --config configs/base.yaml
```

### `configs/base.yaml`

```yaml
model:
  d_model:         128
  n_heads:         8
  encoder_layers:  4
  decoder_layers:  4
  d_ff:            512
  dropout:         0.1
  max_seq_len:     512

vocab:
  move_vocab_size:  1971   # PAD + SOS + EOS + ~1968 UCI moves
  piece_vocab_size: 14     # 12 pieces + EMPTY + CLS

data:
  pgn_dir:         data/pgn/         # directory of .pgn files
  cache_dir:       data/cache/       # tokenized game cache
  max_games:       null              # null = use all available
  train_split:     0.95
  val_split:       0.05

training:
  phase:           1                 # 1 = supervised, 2 = RL self-play
  epochs:          20
  batch_size:      64
  learning_rate:   3e-4
  weight_decay:    1e-2
  grad_clip:       1.0
  warmup_steps:    4000
  log_every:       100               # steps between console logs
  eval_every:      1000              # steps between validation runs
  save_every:      5000              # steps between checkpoints
  checkpoint_dir:  checkpoints/

phase2:
  delta_novelty:   0.05             # novelty bonus weight
  delta_decay:     0.999            # decay per step
  top_k_human:     5                # moves below top-k frequency get novelty bonus
  win_reward:      1.0
  loss_reward:     -1.0
  draw_reward:     0.0
  min_win_rate:    0.55             # new model must beat old by this to replace it
  self_play_games: 100             # games per RL update batch
  pretrained_ckpt: checkpoints/phase1_final.pt
```

### `configs/small.yaml`

For fast iteration and debugging on CPU or a single GPU:

```yaml
# Inherits intent of base.yaml with reduced capacity

model:
  d_model:         64
  n_heads:         4
  encoder_layers:  2
  decoder_layers:  2
  d_ff:            256
  dropout:         0.1
  max_seq_len:     128

data:
  max_games:       10000

training:
  epochs:          5
  batch_size:      16
  learning_rate:   3e-4
  checkpoint_dir:  checkpoints/small/
```

### `configs/phase2.yaml`

```yaml
# Inherits model architecture from a Phase 1 checkpoint
# Only training behaviour changes

training:
  phase:           2
  epochs:          10
  batch_size:      32
  learning_rate:   1e-5             # lower LR for fine-tuning
  checkpoint_dir:  checkpoints/phase2/

phase2:
  pretrained_ckpt: checkpoints/phase1_final.pt
  delta_novelty:   0.05
  delta_decay:     0.999
  min_win_rate:    0.55
  self_play_games: 100
```

### `config.py` — Loading and Validation

```python
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    d_model:        int
    n_heads:        int
    encoder_layers: int
    decoder_layers: int
    d_ff:           int
    dropout:        float
    max_seq_len:    int

@dataclass
class TrainingConfig:
    phase:          int
    epochs:         int
    batch_size:     int
    learning_rate:  float
    weight_decay:   float
    grad_clip:      float
    warmup_steps:   int
    log_every:      int
    eval_every:     int
    save_every:     int
    checkpoint_dir: str

@dataclass
class ChessConfig:
    model:    ModelConfig
    training: TrainingConfig
    # ... vocab, data, phase2 fields

def load_config(path: str) -> ChessConfig:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return ChessConfig(
        model=ModelConfig(**raw['model']),
        training=TrainingConfig(**raw['training']),
        ...
    )
```

### `train.py` — Entry Point

```python
import argparse
from config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)

    if config.training.phase == 1:
        run_phase1(config)
    elif config.training.phase == 2:
        run_phase2(config)

if __name__ == '__main__':
    main()
```

A complete run from scratch:

```bash
# Phase 1 — supervised pretraining
python train.py --config configs/base.yaml

# Phase 2 — RL fine-tuning
python train.py --config configs/phase2.yaml
```

---

## 11. Tensor Shape Reference

| Tensor | Shape | Description |
|---|---|---|
| `board_tokens` | `[B, 64]` | Raw piece token indices per square |
| `board_emb` | `[B, 65, d_model]` | After BoardEmbedding (CLS prepended) |
| `H_n` | `[B, 65, d_model]` | After encoder (full self-attention, EMPTY retained) |
| `move_tokens` | `[B, T]` | Input move sequence incl. SOS |
| `move_emb` | `[B, T, d_model]` | After MoveEmbedding |
| `decoder_out` | `[B, T, d_model]` | After decoder stack |
| `logits` | `[B, T, V]` | Raw scores over move vocabulary |
| `probs` | `[B, T, V]` | After softmax |

---

## 12. Implementation Order

```
Step 1   configs/base.yaml   write base config — model, data, training fields
Step 2   config.py           YAML loader, dataclasses, validation
Step 3   vocab.py            define both vocabularies and index mappings
Step 4   tokenizer.py        board → [64] tokens, game → move token sequence
Step 5   embeddings.py       BoardEmbedding, MoveEmbedding
Step 6   encoder.py          BoardEncoder (fixed [B, 65, d_model] output)
Step 7   decoder.py          MoveDecoder with causal mask and cross-attention
Step 8   model.py            ChessModel assembly and predict_next_move
Step 9   dataset.py          PGNDataset and collate_fn
Step 10  train.py            entry point — reads config, dispatches phase 1 or 2
Step 11  configs/small.yaml  smoke test on small model and small data
Step 12  configs/phase2.yaml RL config pointing to Phase 1 checkpoint
```
