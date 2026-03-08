# Chess Preprocessing Pipeline Design

## 1. Motivation

PGN parsing during training is the primary bottleneck:

```
On-the-fly (slow)               Preprocessed (fast)
──────────────────               ───────────────────
Read .pgn file                   Read .h5 file
Parse PGN text                   Slice pre-baked array
python-chess board replay        Done
Tokenize each square
Tokenize each move
Collate into tensors
↓
GPU finally gets data
```

Preprocessing runs **once**. Training reads pre-baked integer arrays directly from HDF5 — the GPU is never starved waiting for CPU parsing.

---

## 2. What Gets Stored

For every turn `t` in every game, we store:

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `board` | `[64]` | `uint8` | Piece token index per square |
| `moves` | `[T]` | `uint16` | Full move token sequence up to turn t |
| `target` | `scalar` | `uint16` | Next move token (the label) |
| `outcome` | `scalar` | `int8` | +1 win / 0 draw / -1 loss (from perspective of player to move) |
| `turn` | `scalar` | `uint16` | Turn index t within the game |
| `game_id` | `scalar` | `uint32` | Parent game index (for grouping/debugging) |

Each row in the HDF5 file is one training sample — one board state and its associated move history at a specific turn.

---

## 3. Module Overview

```
chess/
├── preprocess/
│   ├── configs/
│   │   └── preprocess.yaml       # All preprocessing config
│   ├── parse.py                  # PGN → list of (board_tokens, move_tokens, outcome)
│   ├── writer.py                 # Streams processed games into HDF5
│   ├── validate.py               # Sanity checks on output HDF5
│   └── preprocess.py             # Entry point — run this once before training
```

---

## 4. Config — `preprocess/configs/preprocess.yaml`

```yaml
input:
  pgn_dir:        data/raw/pgn/        # source .pgn files (can contain multiple files)
  file_pattern:   "*.pgn"             # glob pattern for pgn files
  max_games:      null                 # null = process everything, int = cap for testing

output:
  hdf5_path:      data/processed/chess_dataset.h5
  chunk_size:     1000                 # HDF5 chunk size (rows per chunk, tunes I/O)
  compression:    gzip                 # gzip | lzf | null
  compression_opts: 4                  # compression level (1=fast, 9=small)

processing:
  min_elo:        1500                 # skip games below this Elo (both players)
  max_moves:      512                  # skip games longer than this
  min_moves:      5                    # skip very short/incomplete games
  workers:        8                    # parallel worker processes
  batch_size:     500                  # games processed per worker batch

split:
  train:          0.95
  val:            0.05
  seed:           42                   # for reproducible train/val split
```

---

## 5. Parse — `preprocess/parse.py`

Responsible for converting a single PGN game into a list of training samples.

```python
import chess
import chess.pgn

def parse_game(
    game: chess.pgn.Game,
    move_vocab: dict[str, int],
) -> list[dict] | None:
    """
    Replay a full game turn by turn.
    Returns one sample dict per turn, or None if game is invalid/filtered.

    Each sample:
        {
            'board':   np.ndarray [64]  uint8   piece token indices
            'moves':   np.ndarray [T]   uint16  move token sequence [SOS, m1,...,mt-1]
            'target':  int              uint16  move token at turn t
            'outcome': int              int8    +1 / 0 / -1
            'turn':    int              uint16  turn index
        }
    """
    board    = chess.Board()
    node     = game
    samples  = []
    outcome  = parse_outcome(game)         # +1 / 0 / -1 from perspective of White

    move_tokens = [SOS_IDX]

    for t, move in enumerate(game.mainline_moves()):
        uci = move.uci()

        if uci not in move_vocab:
            return None                    # illegal or unknown move — skip game

        board_tokens = tokenize_board(board)       # [64] uint8
        target       = move_vocab[uci]             # scalar uint16

        # Outcome from perspective of player to move
        player_outcome = outcome if board.turn == chess.WHITE else -outcome

        samples.append({
            'board':   board_tokens,
            'moves':   np.array(move_tokens, dtype=np.uint16),
            'target':  np.uint16(target),
            'outcome': np.int8(player_outcome),
            'turn':    np.uint16(t),
        })

        move_tokens.append(target)
        board.push(move)

    return samples


def tokenize_board(board: chess.Board) -> np.ndarray:
    """
    Returns [64] uint8 array, index 0=a1, index 63=h8.
    Values correspond to PIECE_TOKENS vocab.
    """
    tokens = np.full(64, EMPTY_IDX, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        tokens[square] = PIECE_TOKEN_MAP[(piece.piece_type, piece.color)]
    return tokens


def parse_outcome(game: chess.pgn.Game) -> int:
    """
    Returns +1 if White wins, -1 if Black wins, 0 for draw/unknown.
    """
    result = game.headers.get('Result', '*')
    return {'1-0': 1, '0-1': -1, '1/2-1/2': 0}.get(result, 0)
```

---

## 6. Writer — `preprocess/writer.py`

Streams processed samples into HDF5. Never loads the full dataset into RAM.

```python
import h5py
import numpy as np

class HDF5Writer:
    """
    Streams training samples into a single HDF5 file.
    Uses resizable datasets — no need to know total size upfront.

    File structure:
        chess_dataset.h5
        ├── train/
        │   ├── board       [N_train, 64]  uint8
        │   ├── moves       [N_train, T_max]  uint16  (padded with PAD_IDX)
        │   ├── move_lengths [N_train]      uint16  (actual length before padding)
        │   ├── target      [N_train]       uint16
        │   ├── outcome     [N_train]       int8
        │   ├── turn        [N_train]       uint16
        │   └── game_id     [N_train]       uint32
        └── val/
            └── (same structure)
    """

    def __init__(self, path: str, config: PreprocessConfig):
        self.file   = h5py.File(path, 'w')
        self.config = config
        self._init_datasets()
        self._counts = {'train': 0, 'val': 0}

    def _init_datasets(self):
        for split in ['train', 'val']:
            grp = self.file.create_group(split)
            opts = dict(
                chunks=True,
                maxshape=(None,),
                compression=self.config.compression,
                compression_opts=self.config.compression_opts,
            )
            grp.create_dataset('board',        shape=(0, 64),    dtype='uint8',  maxshape=(None, 64),    **{k:v for k,v in opts.items() if k != 'maxshape'}, chunks=(self.config.chunk_size, 64))
            grp.create_dataset('moves',        shape=(0, MAX_SEQ_LEN), dtype='uint16', maxshape=(None, MAX_SEQ_LEN), chunks=(self.config.chunk_size, MAX_SEQ_LEN))
            grp.create_dataset('move_lengths', shape=(0,),        dtype='uint16', **opts)
            grp.create_dataset('target',       shape=(0,),        dtype='uint16', **opts)
            grp.create_dataset('outcome',      shape=(0,),        dtype='int8',   **opts)
            grp.create_dataset('turn',         shape=(0,),        dtype='uint16', **opts)
            grp.create_dataset('game_id',      shape=(0,),        dtype='uint32', **opts)

    def write_game(self, samples: list[dict], game_id: int, split: str):
        """Append all turns from one game to the appropriate split."""
        n   = len(samples)
        grp = self.file[split]
        cur = self._counts[split]

        # Resize all datasets
        for key in grp:
            if key in ('board', 'moves'):
                continue
            grp[key].resize(cur + n, axis=0)
        grp['board'].resize(cur + n, axis=0)
        grp['moves'].resize(cur + n, axis=0)

        # Pad move sequences to MAX_SEQ_LEN
        moves_padded = np.full((n, MAX_SEQ_LEN), PAD_IDX, dtype=np.uint16)
        for i, s in enumerate(samples):
            L = len(s['moves'])
            moves_padded[i, :L] = s['moves']

        grp['board'][cur:cur+n]        = np.stack([s['board']   for s in samples])
        grp['moves'][cur:cur+n]        = moves_padded
        grp['move_lengths'][cur:cur+n] = [len(s['moves'])       for s in samples]
        grp['target'][cur:cur+n]       = [s['target']           for s in samples]
        grp['outcome'][cur:cur+n]      = [s['outcome']          for s in samples]
        grp['turn'][cur:cur+n]         = [s['turn']             for s in samples]
        grp['game_id'][cur:cur+n]      = game_id

        self._counts[split] += n

    def close(self):
        # Write metadata
        self.file.attrs['total_train'] = self._counts['train']
        self.file.attrs['total_val']   = self._counts['val']
        self.file.attrs['max_seq_len'] = MAX_SEQ_LEN
        self.file.attrs['created']     = str(datetime.now())
        self.file.close()
```

---

## 7. Entry Point — `preprocess/preprocess.py`

```python
import argparse
import glob
import random
import chess.pgn
from multiprocessing import Pool

def process_pgn_file(args) -> list[list[dict]]:
    """Worker function — parses one .pgn file, returns list of game sample lists."""
    pgn_path, config, move_vocab = args
    results = []
    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if not passes_filters(game, config):
                continue
            samples = parse_game(game, move_vocab)
            if samples:
                results.append(samples)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config    = load_preprocess_config(args.config)
    move_vocab = load_move_vocab()
    pgn_files  = glob.glob(f"{config.input.pgn_dir}/{config.input.file_pattern}")

    print(f"Found {len(pgn_files)} PGN files")

    writer = HDF5Writer(config.output.hdf5_path, config)
    rng    = random.Random(config.split.seed)
    game_id = 0

    with Pool(processes=config.processing.workers) as pool:
        worker_args = [(f, config, move_vocab) for f in pgn_files]
        for file_games in pool.imap_unordered(process_pgn_file, worker_args):
            for samples in file_games:
                split = 'train' if rng.random() < config.split.train else 'val'
                writer.write_game(samples, game_id, split)
                game_id += 1

                if game_id % 10_000 == 0:
                    print(f"  Processed {game_id:,} games — "
                          f"train: {writer._counts['train']:,}  "
                          f"val: {writer._counts['val']:,}")

    writer.close()
    print(f"\nDone. {game_id:,} games → {config.output.hdf5_path}")


if __name__ == '__main__':
    main()
```

Run it once:

```bash
python preprocess/preprocess.py --config preprocess/configs/preprocess.yaml
```

---

## 8. Updated Dataset — `dataset.py`

Training now reads directly from HDF5 — no PGN parsing at all.

```python
import h5py
import torch
from torch.utils.data import Dataset

class ChessHDF5Dataset(Dataset):
    """
    Reads pre-baked training samples from HDF5.
    Each __getitem__ is a single board + move history + target.
    """

    def __init__(self, hdf5_path: str, split: str = 'train'):
        self.file  = h5py.File(hdf5_path, 'r')
        self.grp   = self.file[split]
        self.n     = self.file.attrs['total_train' if split == 'train' else 'total_val']

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        length = int(self.grp['move_lengths'][idx])
        return {
            'board':   torch.from_numpy(self.grp['board'][idx].astype('int64')),
            'moves':   torch.from_numpy(self.grp['moves'][idx][:length].astype('int64')),
            'target':  torch.tensor(int(self.grp['target'][idx]), dtype=torch.long),
            'outcome': torch.tensor(int(self.grp['outcome'][idx]), dtype=torch.float),
        }

    def close(self):
        self.file.close()


def collate_fn(batch: list[dict]) -> dict:
    """
    Pads move sequences to longest in batch.
    board is always [64] — no padding needed.
    """
    boards   = torch.stack([s['board']  for s in batch])    # [B, 64]
    targets  = torch.stack([s['target'] for s in batch])    # [B]
    outcomes = torch.stack([s['outcome'] for s in batch])   # [B]

    lengths  = [s['moves'].size(0) for s in batch]
    T_max    = max(lengths)
    moves    = torch.full((len(batch), T_max), PAD_IDX, dtype=torch.long)
    pad_mask = torch.ones(len(batch), T_max, dtype=torch.bool)

    for i, s in enumerate(batch):
        L = lengths[i]
        moves[i, :L]    = s['moves']
        pad_mask[i, :L] = False                              # False = attend, True = ignore

    return {
        'board':    boards,      # [B, 64]
        'moves':    moves,       # [B, T_max]
        'pad_mask': pad_mask,    # [B, T_max]
        'target':   targets,     # [B]
        'outcome':  outcomes,    # [B]
    }
```

---

## 9. Validate — `preprocess/validate.py`

Quick sanity checks to run after preprocessing before training.

```python
def validate_hdf5(path: str):
    """
    Checks:
    - All datasets present in train/ and val/
    - No out-of-range token indices
    - Board tokens all in [0, PIECE_VOCAB_SIZE)
    - Move tokens all in [0, MOVE_VOCAB_SIZE)
    - Outcomes all in {-1, 0, 1}
    - move_lengths consistent with moves content
    - Sample a random game_id and print a readable summary
    """
```

Run after preprocessing:

```bash
python preprocess/validate.py --hdf5 data/processed/chess_dataset.h5
```

---

## 10. Updated base.yaml

The training config now points to the preprocessed HDF5 instead of raw PGN:

```yaml
data:
  hdf5_path:    data/processed/chess_dataset.h5   # preprocessed dataset
  num_workers:  4                                   # DataLoader workers
  prefetch:     2                                   # batches to prefetch
```

---

## 11. Expected Output

For a 1M game dataset (typical Lichess monthly export):

| Metric | Estimate |
|---|---|
| Raw PGN size | ~5 GB |
| HDF5 output size (gzip-4) | ~8–12 GB |
| Preprocessing time (8 workers) | ~30–60 min |
| Training sample count | ~50–80M rows (avg ~60 turns/game) |
| `__getitem__` latency | < 1ms (direct HDF5 slice) |
| PGN parse time per epoch (old) | Hours |
| HDF5 read time per epoch (new) | Minutes |

---

## 12. Implementation Order

```
Step 1   preprocess/configs/preprocess.yaml    write config
Step 2   preprocess/parse.py                   tokenize boards and moves
Step 3   preprocess/writer.py                  HDF5Writer with resizable datasets
Step 4   preprocess/preprocess.py              entry point, multiprocessing pool
Step 5   preprocess/validate.py                sanity checks on output
Step 6   dataset.py                            ChessHDF5Dataset + collate_fn
Step 7   run on small subset                   max_games: 1000, confirm shapes
Step 8   run on full dataset                   production preprocessing
```
