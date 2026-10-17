# TopCat Chess

A small chess engine built around a transformer encoder and a Neural ODE. Given a position, it
predicts a move distribution (policy) and a scalar evaluation (value) for the side to move.
Everything lives in a single file, [`topcat_engine.py`](topcat_engine.py): tokenizer, model,
datasets, training loops (supervised, rules pretraining, value, self-play RL), evaluation and an
interactive command-line game.

## Architecture

```
FEN ──► tokenizer (side-to-move view) ──► 64 square tokens
                                              │  sum of 5 embeddings per square
                                              ▼  (one shared table, vocab = 29)
                            [CLF] + 64 tokens ──► 4 × pre-norm transformer layer ──► LayerNorm
                                │                                                   │
                     CLF vector h0                                   per-square outputs [64, D]
                                │                                                   │
               Neural ODE: dh/dt = f(h), t ∈ [0, 1]                 rules pretraining heads
                                │                                    (legal moves, attacks)
                             LayerNorm
                        ┌───────┴────────┐
                 policy head        value head
               Linear → 4096      MLP → 1 (no tanh)
```

### Input representation

The board is always encoded from the perspective of the side to move: when black is to move the
position is mirrored (`board.mirror()`), so "my" pawns always move up the board and the model never
needs a side-to-move flag. Each of the 64 squares becomes one token whose embedding is the **sum of
five embeddings looked up in a single shared table** (vocabulary of 29 tokens):

| Stream | Values |
|---|---|
| file | `a` … `h` |
| rank | `1` … `8` |
| color | `player`, `opponent`, `empty` |
| piece | `pawn`, `knight`, `bishop`, `rook`, `queen`, `king`, `en_passant` (virtual piece on the en passant square), `empty` |
| castling | `castling` on a rook's home square that still has the castling right, otherwise `empty` |

File and rank embeddings are the only positional information; there is no separate positional
encoding.

### Encoder

A learned classification vector (`clf_embed`) is prepended, giving 65 tokens. They go through
4 pre-norm transformer layers (model width D = 256, 8 attention heads, feed-forward 256 → 512 → 256
with GELU, dropout 0.1) and a final LayerNorm. The CLF output summarizes the position; the 64 square
outputs are used by the rules pretraining heads.

### Neural ODE

The CLF vector `h0` is evolved by a Neural ODE, `dh/dt = f(h)`, where `f` is an MLP
(256 → 256 → 256 → 256 with tanh), integrated from t = 0 to t = 1 with fixed-step RK4
(step 0.1, i.e. 10 steps and 40 evaluations of `f`) using
[torchdiffeq](https://github.com/rtqichen/torchdiffeq). The idea is to give the model extra,
weight-shared "thinking" depth on the global representation before the output heads.

### Output heads

Both heads read the final ODE state after a LayerNorm:

- **Policy**: `Linear(256, 4096)`, one logit per `(from square, to square)` pair in the player's
  frame: `uid = from_square + 64 · to_square`. Promotions are not distinguished; at play time a
  uid is resolved to the first legal move with that from/to, which is the queen promotion.
  Illegal moves are masked out only when choosing a move, not in the training loss.
- **Value**: MLP 256 → 128 → 1 with GELU and no output squashing, from the side to move's point of
  view. It is trained jointly with the policy on game results (`--value-weight`, see below).

### Rules pretraining heads

`PretrainHeads` holds one small MLP (256 → 256 → out) per task, applied independently to every
square output of the encoder, before the Neural ODE. They live outside `TopcatEngine`, so a
pretrained engine checkpoint has exactly the same keys as any other and can be fine-tuned directly.

| Task | Output per square | Target |
|---|---|---|
| `legal_moves` | 64 logits (softmax) | uniform distribution over the legal destination squares of the piece on that square (squares without legal moves are masked) |
| `my_attackers` | 1 logit (sigmoid) | does my piece on this square attack any opponent piece? |
| `their_attackers` | 1 logit (sigmoid) | does the opponent piece on this square attack any of my pieces? |

New tasks only need a target function `board -> (target, mask)` registered in `PRETRAIN_TASKS`.

### Parameters

| Module | Parameters |
|---|---:|
| embeddings (29 × 256) + CLF vector | 7,680 |
| transformer encoder (4 layers) + final LayerNorm | 2,108,928 |
| Neural ODE function | 197,376 |
| pre-head LayerNorm | 512 |
| value head | 33,025 |
| policy head | 1,052,672 |
| **TopcatEngine total** | **3,400,193** |
| rules pretraining heads (3 tasks, separate module) | 214,338 |

## Installation

```bash
pip install torch chess torchdiffeq zstandard
python topcat_engine.py --mode test
```

`optuna` is optional (not used by any working mode). A GPU is used automatically when available.

## Usage

| Mode | What it does |
|---|---|
| `test` | forward pass on the initial position and on a batch of 8, best move and an analysis of `4k3/8/8/8/8/8/8/4KR2 w - - 0 1` |
| `play` | play against the engine in the terminal (`--color white/black`, moves in UCI or SAN) |
| `policy_data` | build a policy dataset from a Lichess `.pgn` / `.pgn.zst` dump (`--pgn`, `-` reads stdin; `--opening-plies`, `--with-stats`) |
| `pretrain_rules` | rules pretraining of the encoder (`--data` FEN JSON and/or `--random-positions N`, `--tasks`, `--workers`) |
| `policy_train` | supervised policy (+ value) training, then a match against a random player (`--value-weight`, `--patience`, `--games 0` skips the match) |
| `eval_train` | value head training on `scores_dataset/ds1.json` |
| `vs_random` | `--games` games with each color against a random player |
| `match` | `--load` against `--opponent`: `--games` games with each color, both sampling with `--temperature` |
| `selfplay` | symmetric actor-critic self-play (experimental, see below) |

Common options: `--load` (checkpoint), `--out`, `--epochs`, `--batch-size`, `--lr`,
`--max-samples`, `--temperature`, `--seed`, `--embed` / `--heads` / `--layers` (model size).

### Reproducing the reference model

```bash
# 500,019 positions from the first 34,879 games of the March 2017 Lichess dump (~2 minutes,
# streamed: no need to download the whole 2 GB file)
curl -s https://database.lichess.org/standard/lichess_db_standard_rated_2017-03.pgn.zst \
  | python topcat_engine.py --mode policy_data --pgn -

# optional: teach the encoder the rules first
python topcat_engine.py --mode pretrain_rules --data policy_dataset.json --random-positions 200000 --workers 4

python topcat_engine.py --mode policy_train --data policy_dataset.json \
  --epochs 50 --batch-size 1024 --lr 2e-4 --load topcat_pretrain.pt   # drop --load without pretraining
```

Dataset recipe: only games with `WhiteElo >= 1500`; positions at plies 6–120 (1-based) with
`ply % 3 == 0`; the FEN is taken before the move; duplicates are dropped; reading stops at the end
of the game in which the dataset reaches 500,000 positions.

Supervised training: 95/5 train/validation split (`random_split`, seed 42), AdamW (weight decay
1e-4), cosine learning-rate schedule, cross-entropy without a legality mask, gradient clipping at 1.0.
Validation accuracy is measured with illegal moves masked out.

Results of the original run with this recipe (no rules pretraining): validation accuracy ≈ 0.26 and,
against a random player (40 + 40 games, T = 0.5), score ≈ 0.65 (W24 D56 L0).

### Improved recipe

```bash
# opening positions + move counts + game results
curl -s https://database.lichess.org/standard/lichess_db_standard_rated_2017-03.pgn.zst \
  | python topcat_engine.py --mode policy_data --pgn - --opening-plies 10 --with-stats --out policy_dataset_v2.json

python topcat_engine.py --mode policy_train --data policy_dataset_v2.json --load topcat_pretrain.pt \
  --epochs 20 --batch-size 1024 --lr 2e-4 --value-weight 1.0 --patience 3 --seed 0
```

- `--opening-plies N` also samples every ply from 1 to N. The reference recipe starts at ply 6, so
  a model trained on it has never seen the first moves of a game.
- `--with-stats` stores, for every position, how often each move was played and the sum of the game
  results from the side to move's point of view: `{FEN: {"moves": {uid: count}, "result": sum}}`.
  The policy target becomes the distribution of played moves (up to the 16 most played) instead of
  the move of the first game that reached the position, which matters for positions that occur in
  many games, such as openings. Validation accuracy counts the most played move as correct.
- `--value-weight w` trains the value head jointly with the policy:
  `loss = policy cross-entropy + w · MSE(value, mean game result)`. Training it separately would
  move the shared encoder away from the policy.
- `--patience N` stops when the validation objective (policy loss + `w` · value MSE) has not
  improved for N epochs.

### Checkpoints and resuming

`policy_train`, `pretrain_rules` and `eval_train` write the full training state after every epoch
(`<out>_state.pt`: model, optimizer, scheduler, history and RNG states). Re-running the same command
with `--resume` continues from the last completed epoch, or does nothing if training already
finished or stopped early. On CPU the result is bit-identical to an uninterrupted run; on GPU it can
differ only by floating-point nondeterminism. Keep the same `--epochs`, since it defines the
learning-rate schedule.

The model itself (`--out`, a plain `state_dict` that `load_model(path)` loads with `strict=True`) is
the **best epoch by validation objective** for `policy_train`, and the latest epoch for
`pretrain_rules` and `eval_train`.

### Results on Kaggle (Tesla T4)

Reference dataset, 5 epochs of rules pretraining (7 min per epoch), then 50 epochs of policy training
(5 min per epoch), about 4 h 45 min in total:

| | val_acc | vs random (40 + 40 games, T = 0.5) |
|---|---|---|
| original run, no rules pretraining | ≈ 0.26 | 0.65 (W24 D56 L0) |
| this code, with rules pretraining | 0.299 (best 0.301) | 0.81 (W51 D28 L1), 0.77 (W43 D37 L0) |

Rules pretraining reached `legal_moves` recall@k 0.988 and F1 0.995 on both attack tasks. The
policy started to overfit after about 20 epochs: validation loss was lowest at epoch 19 (2.787)
and rose to 2.884 while the training loss kept falling. Validation accuracy stayed at about 0.30
from epoch 17 on. These observations led to the improved recipe above; the notebook compares it
with and without rules pretraining.

### Training on Kaggle

[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) runs the improved recipe on a Kaggle
GPU: dataset, rules pretraining, policy + value training, the same training without rules
pretraining (ablation), and an evaluation of both models against a random player and against each
other (`match`). It stops right away without a GPU, stops before the session time limit, resumes in
the next session from the previous run's output and skips finished steps (for example a rules
pretraining restored from an earlier version). Instructions are in the first cell.

## Self-play RL (experimental)

`selfplay` runs symmetric actor-critic self-play: both sides use the model, rewards are the final
result plus a dense material term, and an optional curriculum starts from simple endgames
(`--curriculum 0`: KQ vs K and KR vs K; `1`: adds KQR, KRR, KQ vs KP and KQQ; `2`: half endgames,
half the initial position). The TD(0) target negates the next state's value, because the value is
always given from the side to move's perspective. Starting from the supervised model, self-play has
so far made it weaker (score vs random dropped to about 0.50), so it is not part of the recommended
pipeline.

## Known issues

- **`scores_dataset/ds1.json`** (produced by `create_dataset.py`) stores Stockfish scores from
  **white's** point of view, and each score belongs to the position **before** the random move while
  being keyed by the FEN **after** it. `EvaluationDataset` does not flip the sign for black to move,
  so value training on this file is inconsistent with the mirrored tokenizer.
- **En passant tokens:** `board.fen()` only records the en passant square when a capture is actually
  legal, while a `chess.Board` keeps it after every double pawn push. Training data comes from FENs
  and play uses `Board` objects, so the `en_passant` token appears more often at play time than in
  training.
- **Promotions:** the 4096-move encoding ignores the promotion piece (always resolved as a queen).
- **Openings with the reference recipe:** positions start at ply 6, so its models play odd first
  moves (1.e3 in the Kaggle run). Use `--opening-plies` with `--with-stats`.
- **Tactics:** the policy imitates human moves and does not search; it can miss simple tactics
  (in one test it played f4 instead of a back-rank mate in one).
- **Hyperparameter search:** `--mode hpo` is not implemented. The original version never ran
  because of a `TypeError` that was silently caught.

## Repository layout

| Path | Content |
|---|---|
| `topcat_engine.py` | the engine: tokenizer, model, datasets, training, evaluation, CLI |
| `notebooks/kaggle_train.ipynb` | Kaggle GPU training notebook |
| `scores_dataset/ds1.json` | small Stockfish-scored dataset (3,816 positions) used by `eval_train` |
| `create_dataset.py` | script that generated `ds1.json` (needs a local Stockfish binary) |
