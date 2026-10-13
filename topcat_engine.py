"""
TopCat engine: transformer encoder (CLF token) + Neural ODE for chess.

The TopcatEngine attribute names define the state_dict keys and must stay unchanged so
that existing checkpoints keep loading with strict=True.

Usage:
    python topcat_engine.py --mode test
    python topcat_engine.py --mode play --load topcat_policy.pt --color white
    python topcat_engine.py --mode eval_train            # value head on scores_dataset/ds1.json
    python topcat_engine.py --mode selfplay --load topcat_policy.pt

Reference model (pure supervised):
    python topcat_engine.py --mode policy_data --pgn lichess_db_standard_rated_2017-03.pgn.zst
    python topcat_engine.py --mode policy_train --data policy_dataset.json --resume
    python topcat_engine.py --mode vs_random --load topcat_policy.pt --games 40 --temperature 0.5
"""

import argparse
import io
import json
import math
import os
import random
import sys
import time

import chess
import chess.engine
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    odeint = None
    HAS_TORCHDIFFEQ = False

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

N_MOVES = 4096  # from_sq + 64*to_sq, no promotion piece


# =============================================================================
# 1. Tokenizer
# =============================================================================

VOCAB_TOKENS = (
    ['board']                                               # initial token (legacy)
    + list('abcdefgh12345678')                              # files and ranks
    + ['opponent', 'player']                                # colors
    + ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']  # pieces
    + ['en_passant']                                        # virtual piece on the en passant square
    + ['castling']                                          # castling right, on the rook's square
    + ['empty']                                             # empty square
    + ['clf']                                               # classification token
)


class BoardTokenizer:
    """Tokenizes the board always from the perspective of the side to move."""

    def __init__(self):
        self.tokens = list(VOCAB_TOKENS)
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.reverse_vocab = {i: t for t, i in self.vocab.items()}
        self.vocab_size = len(self.tokens)

        self.piece_dict = {chess.PAWN: 'pawn', chess.KNIGHT: 'knight', chess.BISHOP: 'bishop',
                           chess.ROOK: 'rook', chess.QUEEN: 'queen', chess.KING: 'king'}
        self.color_dict = {False: 'opponent', True: 'player'}
        self._piece_ids = {pt: self.vocab[name] for pt, name in self.piece_dict.items()}

        # files and ranks do not depend on the position
        self._col_ids = torch.tensor([self.vocab[chr(97 + sq % 8)] for sq in range(64)], dtype=torch.long)
        self._row_ids = torch.tensor([self.vocab[str(sq // 8 + 1)] for sq in range(64)], dtype=torch.long)

    def tokenize_board(self, board):
        """board: chess.Board or FEN -> (col, row, color, piece, castling), LongTensors [64]."""
        if isinstance(board, str):
            board = chess.Board(board)

        # pawns always move up the board: after mirroring, WHITE = side to move
        if not board.turn:
            board = board.mirror()

        empty = self.vocab['empty']
        color_ids = [empty] * 64
        piece_ids = [empty] * 64
        castling_ids = [empty] * 64

        for sq, piece in board.piece_map().items():
            piece_ids[sq] = self._piece_ids[piece.piece_type]
            color_ids[sq] = self.vocab[self.color_dict[piece.color == chess.WHITE]]

        # en passant takes precedence over whatever is on the square
        if board.ep_square is not None:
            piece_ids[board.ep_square] = self.vocab['en_passant']
            color_ids[board.ep_square] = self.vocab['opponent']

        if board.has_queenside_castling_rights(chess.WHITE):
            castling_ids[0] = self.vocab['castling']
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_ids[7] = self.vocab['castling']
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_ids[56] = self.vocab['castling']
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_ids[63] = self.vocab['castling']

        return (self._col_ids.clone(), self._row_ids.clone(),
                torch.tensor(color_ids, dtype=torch.long),
                torch.tensor(piece_ids, dtype=torch.long),
                torch.tensor(castling_ids, dtype=torch.long))

    def tokenize_batch(self, boards):
        """List of boards/FENs -> 5 LongTensors [B, 64]."""
        tokenized = [self.tokenize_board(b) for b in boards]
        return tuple(torch.stack(stream) for stream in zip(*tokenized))


# =============================================================================
# 2. Move encoding (4096, no promotion piece)
# =============================================================================

def uci_to_uid(uci, mirror=False):
    from_col = ord(uci[0]) - 97
    from_row = int(uci[1]) - 1
    to_col = ord(uci[2]) - 97
    to_row = int(uci[3]) - 1
    if mirror:
        from_row = 7 - from_row
        to_row = 7 - to_row
    return from_col + 8 * from_row + 64 * to_col + 512 * to_row


def uid_to_uci(uid, mirror=False):
    uid = int(uid)
    from_sq, to_sq = uid % 64, uid // 64
    from_col, from_row = from_sq % 8, from_sq // 8
    to_col, to_row = to_sq % 8, to_sq // 8
    if mirror:
        from_row = 7 - from_row
        to_row = 7 - to_row
    return chr(97 + from_col) + str(from_row + 1) + chr(97 + to_col) + str(to_row + 1)


def legal_move_mask(board, mirror=None):
    """bool[4096] of the legal moves. mirror must be `not board.turn` (the default)."""
    if mirror is None:
        mirror = not board.turn
    mask = torch.zeros(N_MOVES, dtype=torch.bool)
    uids = [uci_to_uid(m.uci()[:4], mirror) for m in board.legal_moves]
    if uids:
        mask[uids] = True
    return mask


def uid_to_move(board, uid):
    """Converts a uid into a legal chess.Move. Promotion: first legal move with that from/to."""
    uci = uid_to_uci(uid, mirror=not board.turn)
    for move in board.legal_moves:
        if move.uci()[:4] == uci:
            return move
    return None


# =============================================================================
# 3. Model
# =============================================================================

class TransformerLayer(nn.Module):
    """Pre-norm layer: attention and FFN with residual connections."""

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ODEFunc(nn.Module):
    """Neural ODE dynamics dh/dt = net(h)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        # zero-init of the last layer (overwritten by TopcatEngine._init_weights)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, h):
        return self.net(h)


class TopcatEngine(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, ffn_dim=512, n_layers=4,
                 ode_hidden=256, dropout=0.1, ode_t_end=1.0):
        super().__init__()
        self.tokenizer = BoardTokenizer()  # not a Module: stays out of the state_dict
        self.embed_dim = embed_dim
        self.ode_t_end = ode_t_end

        # ONE table shared by the 5 streams (col, row, color, piece, castling)
        self.embeddings = nn.Embedding(self.tokenizer.vocab_size, embed_dim)
        self.clf_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.encoder = nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads, ffn_dim, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)

        if HAS_TORCHDIFFEQ:
            self.ode_func = ODEFunc(embed_dim, ode_hidden)
        else:
            print('WARNING: torchdiffeq not installed, model built without the Neural ODE '
                  '(checkpoints with ode_func.* will not load with strict=True).')
            self.ode_func = None

        self.pre_head_norm = nn.LayerNorm(embed_dim)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.policy_head = nn.Linear(embed_dim, N_MOVES)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def encode_tokens(self, col, row, color, piece, castling):
        """Transformer output (before the Neural ODE) for every token:
        [B, 65, D], position 0 = CLF, positions 1..64 = squares a1..h8 from the player's view."""
        x = (self.embeddings(col) + self.embeddings(row) + self.embeddings(color)
             + self.embeddings(piece) + self.embeddings(castling))      # [B, 64, D]
        x = torch.cat([self.clf_embed.expand(x.size(0), -1, -1), x], dim=1)  # [B, 65, D]
        for layer in self.encoder:
            x = layer(x)
        return self.final_norm(x)

    def encode(self, col, row, color, piece, castling):
        return self.encode_tokens(col, row, color, piece, castling)[:, 0, :]  # [B, D]

    def forward(self, col, row, color, piece, castling):
        h0 = self.encode(col, row, color, piece, castling)
        if self.ode_func is not None:
            t = torch.tensor([0.0, self.ode_t_end], device=h0.device, dtype=h0.dtype)
            h_final = odeint(self.ode_func, h0, t, method='rk4', options={'step_size': 0.1})[-1]
        else:
            h_final = h0
        h = self.pre_head_norm(h_final)
        return self.policy_head(h), self.value_head(h)                  # [B, 4096], [B, 1]

    @property
    def device(self):
        return next(self.parameters()).device

    def evaluate_board(self, board):
        """-> (logits[4096] on CPU, value float) for the side to move."""
        was_training = self.training
        self.eval()
        tokens = [t.unsqueeze(0).to(self.device) for t in self.tokenizer.tokenize_board(board)]
        with torch.no_grad():
            logits, value = self(*tokens)
        self.train(was_training)
        return logits[0].cpu(), value.item()

    def best_legal_move(self, board):
        """Argmax of the logits restricted to legal moves (None if there is no legal move)."""
        if isinstance(board, str):
            board = chess.Board(board)
        mask = legal_move_mask(board, not board.turn)
        if not mask.any():
            return None
        logits, _ = self.evaluate_board(board)
        logits = logits.masked_fill(~mask, float('-inf'))
        return uid_to_move(board, logits.argmax().item())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(embed_dim=256, num_heads=8, ffn_dim=512, n_layers=4, ode_hidden=256, dropout=0.1):
    model = TopcatEngine(embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim,
                         n_layers=n_layers, ode_hidden=ode_hidden, dropout=dropout).to(DEVICE)
    n_params = count_parameters(model)
    print(f'TopcatEngine: {n_params:,} parameters ({n_params / 1e6:.2f}M) on {DEVICE}')
    return model


def load_model(path, **kwargs):
    model = build_model(**kwargs)
    model.load_state_dict(torch.load(path, map_location=DEVICE), strict=True)
    print(f'Loaded checkpoint: {path}')
    return model


# =============================================================================
# 4. Datasets, supervised training and per-epoch checkpoints
# =============================================================================

def _tokens_to_device(tokens, device):
    """uint8 tokens [B, 5, 64] -> 5 LongTensors [B, 64] on the device."""
    tokens = tokens.to(device).long()
    return tuple(tokens[:, i] for i in range(5))


def _training_state_path(checkpoint_path):
    base = checkpoint_path[:-3] if checkpoint_path.endswith('.pt') else checkpoint_path
    return base + '_state.pt'


def _save_training_state(path, epoch, modules, optimizer, scheduler, history):
    """Everything needed to resume after `epoch` completed epochs."""
    state = {'epoch': epoch,
             'modules': {name: module.state_dict() for name, module in modules.items()},
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
             'history': history,
             'torch_rng': torch.get_rng_state(),
             'python_rng': random.getstate()}
    if torch.cuda.is_available():
        state['cuda_rng'] = torch.cuda.get_rng_state_all()
    tmp_path = path + '.tmp'
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # atomic: an interrupted save never corrupts the previous state


def _load_training_state(path, modules, optimizer, scheduler):
    """Restores a state written by _save_training_state. -> (completed epochs, history)."""
    state = torch.load(path, map_location='cpu')
    for name, module in modules.items():
        module.load_state_dict(state['modules'][name])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['torch_rng'])
    version, internal, gauss_next = state['python_rng']
    random.setstate((version, tuple(internal), gauss_next))
    if 'cuda_rng' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['epoch'], state['history']


def _maybe_resume(resume, checkpoint_path, n_epochs, modules, optimizer, scheduler, history):
    """With resume=True, continues from <checkpoint>_state.pt if it exists. -> (start_epoch, history)."""
    state_path = _training_state_path(checkpoint_path)
    if not resume:
        return 0, history
    if not os.path.exists(state_path):
        print(f'No training state at {state_path}; starting from scratch')
        return 0, history
    start_epoch, history = _load_training_state(state_path, modules, optimizer, scheduler)
    print(f'Resuming from {state_path}: {start_epoch}/{n_epochs} epochs done')
    return start_epoch, history


class PolicyDataset(torch.utils.data.Dataset):
    """Pre-tokenized JSON {FEN: move_uid}. Tokens are stored as uint8 [N, 5, 64]."""

    def __init__(self, tokenizer, json_path, max_samples=None):
        with open(json_path) as f:
            data = json.load(f)

        self.fens, tokens, targets = [], [], []
        n_invalid = 0
        t0 = time.time()
        for fen, uid in data.items():
            if max_samples is not None and len(self.fens) >= max_samples:
                break
            try:
                uid = int(uid)
                if not 0 <= uid < N_MOVES:
                    raise ValueError(f'uid out of range: {uid}')
                tok = tokenizer.tokenize_board(fen)
            except (ValueError, TypeError):
                n_invalid += 1
                continue
            self.fens.append(fen)
            tokens.append(torch.stack(tok))
            targets.append(uid)
            if len(self.fens) % 100_000 == 0:
                print(f'  tokenized {len(self.fens):,} positions ({time.time() - t0:.0f}s)', flush=True)

        self.tokens = torch.stack(tokens).to(torch.uint8)
        self.targets = torch.tensor(targets, dtype=torch.long)
        print(f'PolicyDataset: {len(self.fens):,} positions ({n_invalid} invalid skipped) from {json_path}')

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        tok = self.tokens[idx].long()
        return tok[0], tok[1], tok[2], tok[3], tok[4], self.targets[idx]

    def get_batch(self, idx, device=DEVICE):
        return _tokens_to_device(self.tokens[idx], device), self.targets[idx].to(device)


def _split_indices(dataset, val_fraction, seed=42):
    n = len(dataset)
    n_train = int(round((1.0 - val_fraction) * n))
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(seed))
    return torch.tensor(train_set.indices, dtype=torch.long), torch.tensor(val_set.indices, dtype=torch.long)


def train_policy_supervised(model, json_path, n_epochs=5, batch_size=512, lr=1e-4,
                            max_samples=None, checkpoint_path='topcat_policy.pt', resume=False):
    """Cross-entropy on the policy head. After every epoch the model is written to
    checkpoint_path and the full training state to <checkpoint>_state.pt (see resume)."""
    dataset = PolicyDataset(model.tokenizer, json_path, max_samples)
    train_idx, val_idx = _split_indices(dataset, 0.05)
    device = model.device
    print(f'Train: {len(train_idx):,} | validation: {len(val_idx):,}')

    # legality masks of the validation set (accuracy only)
    val_masks = torch.stack([legal_move_mask(chess.Board(dataset.fens[i])) for i in val_idx.tolist()])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    modules = {'model': model}
    start_epoch, history = _maybe_resume(resume, checkpoint_path, n_epochs, modules, optimizer, scheduler,
                                         {'train_loss': [], 'val_loss': [], 'val_acc': []})

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        loss_sum, n_seen = 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            inputs, target = dataset.get_batch(idx, device)
            logits, _ = model(*inputs)
            loss = criterion(logits, target)  # no legality mask
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(idx)
            n_seen += len(idx)
        scheduler.step()

        model.eval()
        val_loss_sum, n_correct = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                idx = val_idx[start:start + batch_size]
                inputs, target = dataset.get_batch(idx, device)
                logits, _ = model(*inputs)
                val_loss_sum += F.cross_entropy(logits, target, reduction='sum').item()
                mask = val_masks[start:start + batch_size].to(device)
                pred = logits.masked_fill(~mask, float('-inf')).argmax(dim=-1)
                n_correct += (pred == target).sum().item()

        train_loss = loss_sum / max(n_seen, 1)
        val_loss = val_loss_sum / max(len(val_idx), 1)
        val_acc = n_correct / max(len(val_idx), 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f'Epoch {epoch + 1}/{n_epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} '
              f'| val_acc {val_acc:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s',
              flush=True)
        torch.save(model.state_dict(), checkpoint_path)
        _save_training_state(_training_state_path(checkpoint_path), epoch + 1, modules,
                             optimizer, scheduler, history)

    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')
    return history


class EvaluationDataset(torch.utils.data.Dataset):
    """JSON {FEN: centipawns} -> target tanh(cp/400). The cp sign is NOT flipped when black is
    to move, although the tokenizer mirrors the board (see known issues in the README)."""

    def __init__(self, tokenizer, json_path, max_samples=None):
        with open(json_path) as f:
            data = json.load(f)

        self.fens, tokens, targets = [], [], []
        n_invalid = 0
        for fen, cp in data.items():
            if max_samples is not None and len(self.fens) >= max_samples:
                break
            try:
                cp = float(cp)
                tok = tokenizer.tokenize_board(fen)
            except (ValueError, TypeError):
                n_invalid += 1
                continue
            self.fens.append(fen)
            tokens.append(torch.stack(tok))
            targets.append(math.tanh(cp / 400.0))

        self.tokens = torch.stack(tokens).to(torch.uint8)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        print(f'EvaluationDataset: {len(self.fens):,} positions ({n_invalid} invalid skipped) from {json_path}')

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        tok = self.tokens[idx].long()
        return tok[0], tok[1], tok[2], tok[3], tok[4], self.targets[idx]

    def get_batch(self, idx, device=DEVICE):
        return _tokens_to_device(self.tokens[idx], device), self.targets[idx].to(device)


def train_evaluation(model, json_path='scores_dataset/ds1.json', n_epochs=20, batch_size=256,
                     lr=1e-4, val_split=0.1, max_samples=None, checkpoint_path='topcat_eval.pt',
                     resume=False):
    """MSE on the value head, with the same per-epoch checkpoints as train_policy_supervised."""
    dataset = EvaluationDataset(model.tokenizer, json_path, max_samples)
    train_idx, val_idx = _split_indices(dataset, val_split)
    device = model.device

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    modules = {'model': model}
    start_epoch, history = _maybe_resume(resume, checkpoint_path, n_epochs, modules, optimizer, scheduler,
                                         {'train_loss': [], 'val_loss': []})

    for epoch in range(start_epoch, n_epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        loss_sum = 0.0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            inputs, target = dataset.get_batch(idx, device)
            _, value = model(*inputs)
            loss = F.mse_loss(value.squeeze(-1), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(idx)
        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                idx = val_idx[start:start + batch_size]
                inputs, target = dataset.get_batch(idx, device)
                _, value = model(*inputs)
                val_loss_sum += F.mse_loss(value.squeeze(-1), target, reduction='sum').item()

        train_loss = loss_sum / max(len(train_idx), 1)
        val_loss = val_loss_sum / max(len(val_idx), 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch + 1}/{n_epochs} | train_mse {train_loss:.4f} | val_mse {val_loss:.4f}', flush=True)
        torch.save(model.state_dict(), checkpoint_path)
        _save_training_state(_training_state_path(checkpoint_path), epoch + 1, modules,
                             optimizer, scheduler, history)

    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')
    return history


# =============================================================================
# 4b. Policy dataset from Lichess PGN dumps (.pgn or .pgn.zst)
# =============================================================================

def _is_zstd(magic):
    # regular frame (28 B5 2F FD) or skippable frame (5? 2A 4D 18), used by the Lichess dumps
    return magic == b'\x28\xb5\x2f\xfd' or (len(magic) == 4 and magic[1:] == b'\x2a\x4d\x18'
                                            and 0x50 <= magic[0] <= 0x5f)


def _open_pgn(path):
    """Opens .pgn or .pgn.zst as text. path='-' reads stdin (e.g. curl ... | python ...)."""
    if path == '-':
        raw = sys.stdin.buffer
        is_zstd = _is_zstd(raw.peek(4)[:4])
    else:
        raw = open(path, 'rb')
        is_zstd = _is_zstd(raw.read(4))
        raw.seek(0)
    if is_zstd:
        import zstandard
        raw = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
    return io.TextIOWrapper(raw, encoding='utf-8', errors='replace')


def build_policy_dataset_from_pgn(pgn_path, out_json='policy_dataset.json', min_elo=1500,
                                  min_ply=6, max_ply=120, ply_step=3, max_positions=500_000):
    """Builds {FEN: uid} with the reference recipe: only games with WhiteElo >= min_elo,
    ply (1-based: number of the half-move about to be played) in min_ply..max_ply with
    ply % ply_step == 0, FEN before the move, deduplicated by FEN. Stops at the end of the game
    in which the total reaches max_positions.
    lichess_db_standard_rated_2017-03 yields 500,019 positions from 34,879 games read."""
    data = {}
    n_read = n_used = 0
    t0 = time.time()
    with _open_pgn(pgn_path) as pgn:
        while len(data) < max_positions:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            n_read += 1
            if n_read % 5000 == 0:
                print(f'  {n_read:,} games read | {len(data):,} positions ({time.time() - t0:.0f}s)', flush=True)
            try:
                white_elo = int(game.headers.get('WhiteElo', ''))
            except ValueError:
                continue
            if white_elo < min_elo:
                continue
            n_used += 1
            board = game.board()
            for move in game.mainline_moves():
                ply = board.ply() + 1
                if ply > max_ply:
                    break
                if ply >= min_ply and ply % ply_step == 0:
                    fen = board.fen()
                    if fen not in data:
                        data[fen] = uci_to_uid(move.uci()[:4], mirror=not board.turn)
                board.push(move)

    with open(out_json, 'w') as f:
        json.dump(data, f)
    print(f'{len(data):,} positions from {n_read:,} games read ({n_used:,} with WhiteElo >= {min_elo}) '
          f'saved to {out_json}')
    return data


# =============================================================================
# 4c. Rules pretraining: per-square heads on the transformer output
# =============================================================================
#
# Each task receives the board already from the side to move's perspective (WHITE = player,
# same frame as the tokens) and returns (target [64, out_dim] bool, mask [64] bool).
# The mask selects the squares that enter the loss and the metrics. To add a task, write
# its target function and register it in PRETRAIN_TASKS.
#   kind 'dist':   softmax over out_dim; target = uniform distribution over the True entries
#   kind 'binary': per-square sigmoid (out_dim = 1)

def _target_legal_moves(board):
    """Legal destinations from each square; only squares with a legal move enter the loss."""
    target = torch.zeros(64, 64, dtype=torch.bool)
    moves = list(board.legal_moves)
    if moves:
        target[[m.from_square for m in moves], [m.to_square for m in moves]] = True
    return target, target.any(dim=-1)


def _attack_flags(board, attacker, victim):
    """For each `attacker` piece: does it attack any `victim` piece? (attacks as in
    board.attacks: pins are ignored, pawns attack diagonally only)."""
    target = torch.zeros(64, 1, dtype=torch.bool)
    mask = torch.zeros(64, dtype=torch.bool)
    victims = board.occupied_co[victim]
    for sq in chess.scan_forward(board.occupied_co[attacker]):
        mask[sq] = True
        target[sq, 0] = bool(board.attacks_mask(sq) & victims)
    return target, mask


def _target_my_attackers(board):
    """Squares holding one of my pieces that attacks an opponent piece."""
    return _attack_flags(board, chess.WHITE, chess.BLACK)


def _target_their_attackers(board):
    """Squares holding an opponent piece that attacks one of my pieces."""
    return _attack_flags(board, chess.BLACK, chess.WHITE)


PRETRAIN_TASKS = {
    'legal_moves': {'out_dim': 64, 'kind': 'dist', 'target': _target_legal_moves},
    'my_attackers': {'out_dim': 1, 'kind': 'binary', 'target': _target_my_attackers},
    'their_attackers': {'out_dim': 1, 'kind': 'binary', 'target': _target_their_attackers},
}
DEFAULT_PRETRAIN_TASKS = ('legal_moves', 'my_attackers', 'their_attackers')


class PretrainHeads(nn.Module):
    """One MLP per task, applied to every square of the transformer output (before the Neural
    ODE). Kept outside TopcatEngine so the engine's state_dict keys do not change."""

    def __init__(self, embed_dim, tasks=DEFAULT_PRETRAIN_TASKS, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.tasks = list(tasks)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                                nn.Linear(hidden_dim, PRETRAIN_TASKS[name]['out_dim']))
            for name in self.tasks})

    def forward(self, squares):
        """squares: [B, 64, D] -> {task: logits [B, 64, out_dim]}"""
        return {name: head(squares) for name, head in self.heads.items()}


def load_pretrain_heads(path, embed_dim=256):
    saved = torch.load(path, map_location=DEVICE)
    heads = PretrainHeads(embed_dim, saved['tasks']).to(DEVICE)
    heads.load_state_dict(saved['state_dict'])
    return heads


class RulesDataset(torch.utils.data.Dataset):
    """FENs -> (uint8 tokens [5, 64], {task: target}, {task: mask}); targets computed on the fly."""

    def __init__(self, tokenizer, fens, tasks=DEFAULT_PRETRAIN_TASKS):
        self.tokenizer = tokenizer
        self.fens = list(fens)
        self.tasks = list(tasks)

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        board = chess.Board(self.fens[idx])
        tokens = torch.stack(self.tokenizer.tokenize_board(board)).to(torch.uint8)
        board = board if board.turn == chess.WHITE else board.mirror()  # same frame as the tokens
        targets, masks = {}, {}
        for name in self.tasks:
            targets[name], masks[name] = PRETRAIN_TASKS[name]['target'](board)
        return tokens, targets, masks


def load_fens(json_path):
    """Valid FENs from the keys of a JSON {FEN: ...} (e.g. policy_dataset.json, ds1.json) or a list."""
    with open(json_path) as f:
        data = json.load(f)
    fens = []
    for fen in data:
        try:
            chess.Board(fen)
        except ValueError:
            continue
        fens.append(fen)
    print(f'{len(fens):,} FENs from {json_path}')
    return fens


def generate_random_fens(n_positions, max_plies=200, seed=None):
    """Positions from games played with random moves (no duplicates)."""
    rng = random.Random(seed)
    fens = {}
    while len(fens) < n_positions:
        board = chess.Board()
        while len(fens) < n_positions and not board.is_game_over() and board.ply() < max_plies:
            board.push(rng.choice(list(board.legal_moves)))
            fens.setdefault(board.fen(), None)
    print(f'{len(fens):,} random positions generated')
    return list(fens)


def _pretrain_loss(kind, logits, target, mask):
    mask = mask.float()
    if kind == 'dist':
        target = target.float()
        probs = target / target.sum(dim=-1, keepdim=True).clamp(min=1)
        per_square = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    else:
        per_square = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), target.squeeze(-1).float(), reduction='none')
    return (per_square * mask).sum() / mask.sum().clamp(min=1)


def _pretrain_counts(kind, logits, target, mask):
    """Counts that can be summed across batches for the validation metrics."""
    if kind == 'dist':
        # recall@k: of the k most likely destinations (k = number of legal ones), how many are legal
        k = target.sum(dim=-1)
        ranks = logits.argsort(dim=-1, descending=True).argsort(dim=-1)
        recall = (target & (ranks < k.unsqueeze(-1))).sum(dim=-1).float() / k.clamp(min=1)
        top1 = target.gather(-1, logits.argmax(dim=-1, keepdim=True)).squeeze(-1)
        return {'n': mask.sum().item(), 'recall@k': recall[mask].sum().item(), 'top1': top1[mask].sum().item()}
    pred = logits.squeeze(-1) > 0
    target = target.squeeze(-1)
    return {'n': mask.sum().item(), 'acc': (pred == target)[mask].sum().item(),
            'tp': (pred & target)[mask].sum().item(), 'fp': (pred & ~target)[mask].sum().item(),
            'fn': (~pred & target)[mask].sum().item()}


def _pretrain_metrics(kind, counts):
    n = max(counts['n'], 1)
    if kind == 'dist':
        return {'recall@k': counts['recall@k'] / n, 'top1': counts['top1'] / n}
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    return {'acc': counts['acc'] / n, 'f1': 2 * tp / max(2 * tp + fp + fn, 1)}


def pretrain_rules(model, fens, tasks=DEFAULT_PRETRAIN_TASKS, n_epochs=5, batch_size=512, lr=3e-4,
                   task_weights=None, val_fraction=0.05, num_workers=0, heads=None,
                   checkpoint_path='topcat_pretrain.pt', heads_path='topcat_pretrain_heads.pt',
                   resume=False):
    """Trains embeddings + transformer (not the ODE/policy/value) with the auxiliary heads.
    After every epoch it writes the engine state_dict (loads with strict=True, e.g.
    --mode policy_train --load), the heads (separate file) and <checkpoint>_state.pt for resume.
    -> (history, heads)."""
    tasks = list(tasks)
    device = model.device
    if heads is None:
        heads = PretrainHeads(model.embed_dim, tasks).to(device)
    weights = {name: 1.0 for name in tasks}
    weights.update(task_weights or {})

    dataset = RulesDataset(model.tokenizer, fens, tasks)
    train_idx, val_idx = _split_indices(dataset, val_fraction)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_idx.tolist()), batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_idx.tolist()), batch_size=batch_size,
        num_workers=num_workers)
    print(f'Pretraining {tasks} | train: {len(train_idx):,} | validation: {len(val_idx):,}')

    params = ([model.clf_embed] + list(model.embeddings.parameters()) + list(model.encoder.parameters())
              + list(model.final_norm.parameters()) + list(heads.parameters()))
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    modules = {'model': model, 'heads': heads}
    start_epoch, history = _maybe_resume(resume, checkpoint_path, n_epochs, modules, optimizer, scheduler,
                                         {'train_loss': [], 'val_loss': [], 'val_metrics': []})

    def run_batch(tokens, targets, masks):
        squares = model.encode_tokens(*_tokens_to_device(tokens, device))[:, 1:, :]
        outputs = heads(squares)
        losses = {name: _pretrain_loss(PRETRAIN_TASKS[name]['kind'], outputs[name],
                                       targets[name].to(device), masks[name].to(device))
                  for name in tasks}
        return outputs, losses

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        model.train()
        heads.train()
        train_sum = {name: 0.0 for name in tasks}
        n_batches = 0
        for tokens, targets, masks in train_loader:
            _, losses = run_batch(tokens, targets, masks)
            loss = sum(weights[name] * losses[name] for name in tasks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            for name in tasks:
                train_sum[name] += losses[name].item()
            n_batches += 1
        scheduler.step()

        model.eval()
        heads.eval()
        val_sum = {name: 0.0 for name in tasks}
        counts = {name: {} for name in tasks}
        n_val_batches = 0
        with torch.no_grad():
            for tokens, targets, masks in val_loader:
                outputs, losses = run_batch(tokens, targets, masks)
                for name in tasks:
                    val_sum[name] += losses[name].item()
                    batch_counts = _pretrain_counts(PRETRAIN_TASKS[name]['kind'], outputs[name],
                                                    targets[name].to(device), masks[name].to(device))
                    for key, value in batch_counts.items():
                        counts[name][key] = counts[name].get(key, 0) + value
                n_val_batches += 1

        train_loss = {name: train_sum[name] / max(n_batches, 1) for name in tasks}
        val_loss = {name: val_sum[name] / max(n_val_batches, 1) for name in tasks}
        metrics = {name: _pretrain_metrics(PRETRAIN_TASKS[name]['kind'], counts[name]) for name in tasks}
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(metrics)
        print(f'Epoch {epoch + 1}/{n_epochs} ({time.time() - t0:.0f}s)', flush=True)
        for name in tasks:
            metric_text = ' | '.join(f'{k} {v:.4f}' for k, v in metrics[name].items())
            print(f'  {name:16s} train {train_loss[name]:.4f} | val {val_loss[name]:.4f} | {metric_text}',
                  flush=True)
        torch.save(model.state_dict(), checkpoint_path)
        torch.save({'tasks': tasks, 'state_dict': heads.state_dict()}, heads_path)
        _save_training_state(_training_state_path(checkpoint_path), epoch + 1, modules,
                             optimizer, scheduler, history)

    torch.save(model.state_dict(), checkpoint_path)
    torch.save({'tasks': tasks, 'state_dict': heads.state_dict()}, heads_path)
    print(f'Engine saved to {checkpoint_path} | heads to {heads_path}')
    return history, heads


# =============================================================================
# 5. RL (symmetric actor-critic self-play) — not the path to the best model
# =============================================================================

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _material_balance(board):
    """(white - black) / 20, clamped to [-1, 1]."""
    balance = sum(v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
                  for pt, v in _PIECE_VALUES.items())
    return max(-1.0, min(1.0, balance / 20.0))


def _select_move_stochastic(board, logits, temperature=1.0):
    """Samples a move from softmax(legal logits / T). -> (move, log_prob, probs)."""
    mirror = not board.turn
    moves, uids, seen = [], [], set()
    for move in board.legal_moves:
        uid = uci_to_uid(move.uci()[:4], mirror)
        if uid in seen:  # promotions: keep the first legal move with that from/to
            continue
        seen.add(uid)
        moves.append(move)
        uids.append(uid)
    if not moves:
        return None, None, None
    probs = F.softmax(logits[uids] / max(temperature, 1e-6), dim=-1)
    i = torch.multinomial(probs, 1).item()
    return moves[i], torch.log(probs[i]), probs


# specs: (pieces of the strong side, pieces of the weak side)
_CURRICULUM_EASY = [('KQ', 'K'), ('KR', 'K')]
_CURRICULUM_MEDIUM = [('KQR', 'K'), ('KRR', 'K'), ('KQ', 'KP'), ('KQQ', 'K')]

_PIECE_LETTERS = {'K': chess.KING, 'Q': chess.QUEEN, 'R': chess.ROOK,
                  'B': chess.BISHOP, 'N': chess.KNIGHT, 'P': chess.PAWN}


def _random_endgame_board(strong, weak):
    """Random valid, unfinished position; strong side color and side to move are random."""
    while True:
        board = chess.Board(None)
        strong_color = random.choice([chess.WHITE, chess.BLACK])
        free = list(range(64))
        random.shuffle(free)
        ok = True
        for pieces, color in ((strong, strong_color), (weak, not strong_color)):
            for letter in pieces:
                piece_type = _PIECE_LETTERS[letter]
                for k, sq in enumerate(free):
                    if piece_type != chess.PAWN or 1 <= chess.square_rank(sq) <= 6:
                        board.set_piece_at(free.pop(k), chess.Piece(piece_type, color))
                        break
                else:
                    ok = False
        board.turn = random.choice([chess.WHITE, chess.BLACK])
        if ok and board.is_valid() and not board.is_game_over():
            return board


def _curriculum_board(level):
    """None: initial position; 0: easy; 1: easy + medium; 2: 50% endgame / 50% initial position."""
    if level is None:
        return chess.Board()
    if level == 0:
        specs = _CURRICULUM_EASY
    elif level == 1:
        specs = _CURRICULUM_EASY + _CURRICULUM_MEDIUM
    else:
        if random.random() < 0.5:
            return chess.Board()
        specs = _CURRICULUM_EASY + _CURRICULUM_MEDIUM
    return _random_endgame_board(*random.choice(specs))


def self_play_symmetric_episode(model, max_moves=120, temperature=1.0, material_coef=0.1,
                                temp_white=None, temp_black=None, curriculum_level=None):
    """Both sides play with the model. Dense reward = material_coef * Δmaterial from the mover's
    point of view. -> (trajectory, result ±1/0 from white's point of view,
    bootstrap_value = V(s_final) if the game was truncated, else 0)."""
    temp_white = temperature if temp_white is None else temp_white
    temp_black = temperature if temp_black is None else temp_black
    board = _curriculum_board(curriculum_level)
    tokenizer = model.tokenizer
    device = model.device

    was_training = model.training
    model.eval()
    trajectory = []
    with torch.no_grad():
        while not board.is_game_over() and len(trajectory) < max_moves:
            tokens = tokenizer.tokenize_board(board)
            logits, value = model(*[t.unsqueeze(0).to(device) for t in tokens])
            T = temp_white if board.turn == chess.WHITE else temp_black
            move, log_prob, _ = _select_move_stochastic(board, logits[0].cpu(), T)
            player = 1.0 if board.turn == chess.WHITE else -1.0
            mirror = not board.turn
            mask = legal_move_mask(board, mirror)
            material_before = _material_balance(board)
            board.push(move)
            reward = material_coef * (_material_balance(board) - material_before) * player
            trajectory.append({'tokens': torch.stack(tokens).to(torch.uint8),
                               'action': uci_to_uid(move.uci()[:4], mirror),
                               'mask': mask, 'player': player, 'reward': reward,
                               'log_prob': log_prob.item(), 'value': value.item()})

        outcome = board.outcome()
        if outcome is not None:
            result = 0.0 if outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else -1.0)
            bootstrap_value = 0.0
        else:
            result = 0.0
            tokens = tokenizer.tokenize_board(board)
            _, value = model(*[t.unsqueeze(0).to(device) for t in tokens])
            bootstrap_value = value.item()
    model.train(was_training)
    return trajectory, result, bootstrap_value


def actor_critic_update(model, optimizer, episodes, gamma=0.99, value_coef=0.5,
                        entropy_coef=0.01, max_grad_norm=1.0):
    """TD(0) actor-critic over a list of episodes (trajectory, result, bootstrap_value).

    V(s) is always from the side to move's point of view (the tokenizer mirrors the board).
    Since the turn alternates every move, V(s_{t+1}) seen by whoever moved at t is -V(s_{t+1}):
    G_t = r_t + γ·(-V(s_{t+1})), with r_T += result * player and a bootstrap at the last state.
    """
    episodes = [e for e in episodes if len(e[0]) > 0]
    if not episodes:
        return None
    device = model.device

    tokens, actions, masks, rewards, lengths, bootstraps = [], [], [], [], [], []
    for trajectory, result, bootstrap_value in episodes:
        r = [step['reward'] for step in trajectory]
        r[-1] += result * trajectory[-1]['player']
        tokens += [step['tokens'] for step in trajectory]
        actions += [step['action'] for step in trajectory]
        masks += [step['mask'] for step in trajectory]
        rewards += r
        lengths.append(len(trajectory))
        bootstraps.append(bootstrap_value)

    inputs = _tokens_to_device(torch.stack(tokens), device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    masks = torch.stack(masks).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

    model.train()
    logits, values = model(*inputs)
    values = values.squeeze(-1)

    # raw V(s_{t+1}) (next player's view) with a bootstrap at the end of each episode
    with torch.no_grad():
        next_values = torch.empty_like(values)
        start = 0
        for length, bootstrap_value in zip(lengths, bootstraps):
            end = start + length
            next_values[start:end - 1] = values[start + 1:end]
            next_values[end - 1] = bootstrap_value
            start = end
        returns = rewards - gamma * next_values
        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_probs = F.log_softmax(logits.masked_fill(~masks, -1e9), dim=-1)
    log_prob_actions = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

    policy_loss = -(log_prob_actions * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return {'loss': loss.item(), 'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(), 'entropy': entropy.item()}


def run_selfplay_symmetric_loop(model, n_episodes=200, n_batch=4, lr=1e-5, gamma=0.99,
                                value_coef=0.5, entropy_coef=0.01, temperatures=(0.8, 1.2),
                                material_coef=0.1, max_moves=120, curriculum_level=0,
                                eval_every=50, eval_games=10, checkpoint_path='topcat_selfplay.pt'):
    """Symmetric self-play; white/black temperatures swap every episode."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_episodes // n_batch))
    history = {'result': [], 'length': [], 'loss': [], 'entropy': [], 'vs_random': []}
    batch = []

    for episode in range(n_episodes):
        t_a, t_b = temperatures
        temp_white, temp_black = (t_a, t_b) if episode % 2 == 0 else (t_b, t_a)
        trajectory, result, bootstrap_value = self_play_symmetric_episode(
            model, max_moves=max_moves, material_coef=material_coef, temp_white=temp_white,
            temp_black=temp_black, curriculum_level=curriculum_level)
        batch.append((trajectory, result, bootstrap_value))
        history['result'].append(result)
        history['length'].append(len(trajectory))

        if len(batch) == n_batch:
            stats = actor_critic_update(model, optimizer, batch, gamma=gamma,
                                        value_coef=value_coef, entropy_coef=entropy_coef)
            scheduler.step()
            batch = []
            if stats is not None:
                history['loss'].append(stats['loss'])
                history['entropy'].append(stats['entropy'])
                print(f'Episode {episode + 1}/{n_episodes} | result {result:+.0f} | '
                      f'plies {len(trajectory)} | loss {stats["loss"]:.4f} | '
                      f'entropy {stats["entropy"]:.3f}', flush=True)

        if eval_every and (episode + 1) % eval_every == 0:
            history['vs_random'].append(evaluate_vs_random(model, n_games=eval_games, temperature=0.5)['score'])

    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')
    return history


def run_hpo(*args, **kwargs):
    raise NotImplementedError(
        'The original HPO never worked: train_self_play / run_selfplay_rl_loop called '
        'self_play_episode with nonexistent kwargs (temp_exploit, use_random_opponent) and the '
        'TypeError was swallowed by catch=(Exception,). It was not recreated; define the objective '
        '(e.g. score vs random after run_selfplay_symmetric_loop) to implement it.')


# =============================================================================
# 6. Evaluation and interactive play
# =============================================================================

def _model_move(model, board, temperature=0.0):
    if temperature <= 0:
        return model.best_legal_move(board)
    logits, _ = model.evaluate_board(board)
    move, _, _ = _select_move_stochastic(board, logits, temperature)
    return move


def analyze_position(model, board, top_k=10):
    """Prints the top_k legal moves (probability among legal moves) and the side to move's value."""
    if isinstance(board, str):
        board = chess.Board(board)
    logits, value = model.evaluate_board(board)
    mirror = not board.turn
    moves, uids, seen = [], [], set()
    for move in board.legal_moves:
        uid = uci_to_uid(move.uci()[:4], mirror)
        if uid not in seen:
            seen.add(uid)
            moves.append(move)
            uids.append(uid)
    probs = F.softmax(logits[uids], dim=-1)
    order = probs.argsort(descending=True)[:top_k].tolist()
    print(board)
    print(f'FEN: {board.fen()}')
    print(f'Value (side to move): {value:+.4f} | legal moves: {len(moves)}')
    for rank, i in enumerate(order, 1):
        print(f'  {rank:2d}. {board.san(moves[i]):8s} {moves[i].uci():6s} '
              f'logit {logits[uids[i]].item():+8.3f}  prob {probs[i].item():.4f}')
    best = model.best_legal_move(board)
    print(f'Best move: {board.san(best)} ({best.uci()})')
    return best


def evaluate_vs_random(model, n_games=20, max_moves=200, temperature=0.5, verbose=True):
    """Plays n_games with each color against a random opponent. score = (W + 0.5·D) / total."""
    wins = draws = losses = 0
    for model_color in (chess.WHITE, chess.BLACK):
        for _ in range(n_games):
            board = chess.Board()
            while not board.is_game_over() and board.ply() < max_moves:
                if board.turn == model_color:
                    move = _model_move(model, board, temperature)
                else:
                    move = random.choice(list(board.legal_moves))
                board.push(move)
            outcome = board.outcome()
            if outcome is None or outcome.winner is None:
                draws += 1
            elif outcome.winner == model_color:
                wins += 1
            else:
                losses += 1
    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total
    if verbose:
        print(f'vs random ({n_games}+{n_games} games, T={temperature}): '
              f'W{wins} D{draws} L{losses} | score {score:.3f}', flush=True)
    return {'wins': wins, 'draws': draws, 'losses': losses, 'score': score}


def evaluate_vs_stockfish(model, engine_path='stockfish', n_games=10, time_limit=0.05,
                          skill_level=None, max_moves=200, temperature=0.0, verbose=True):
    """Plays n_games with each color against Stockfish (optional, needs the executable)."""
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    if skill_level is not None:
        engine.configure({'Skill Level': skill_level})
    wins = draws = losses = 0
    try:
        for model_color in (chess.WHITE, chess.BLACK):
            for _ in range(n_games):
                board = chess.Board()
                while not board.is_game_over() and board.ply() < max_moves:
                    if board.turn == model_color:
                        move = _model_move(model, board, temperature)
                    else:
                        move = engine.play(board, chess.engine.Limit(time=time_limit)).move
                    board.push(move)
                outcome = board.outcome()
                if outcome is None or outcome.winner is None:
                    draws += 1
                elif outcome.winner == model_color:
                    wins += 1
                else:
                    losses += 1
    finally:
        engine.quit()
    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total
    if verbose:
        print(f'vs Stockfish ({n_games}+{n_games} games): W{wins} D{draws} L{losses} | score {score:.3f}')
    return {'wins': wins, 'draws': draws, 'losses': losses, 'score': score}


def _board_ascii(board, orientation=chess.WHITE):
    ranks = range(7, -1, -1) if orientation == chess.WHITE else range(8)
    files = range(8) if orientation == chess.WHITE else range(7, -1, -1)
    lines = []
    for rank in ranks:
        cells = []
        for file in files:
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece.symbol() if piece else '.')
        lines.append(f'{rank + 1}  ' + ' '.join(cells))
    lines.append('   ' + ' '.join(chess.FILE_NAMES[f] for f in files))
    return '\n'.join(lines)


def play_interactive(model, human_color='white', temperature=0.0):
    """Human vs TopCat. Moves in UCI (e2e4) or SAN (Nf3); 'quit' to leave."""
    board = chess.Board()
    human = chess.WHITE if human_color.lower().startswith('w') else chess.BLACK
    print(f'You play {"white" if human == chess.WHITE else "black"}.')
    while not board.is_game_over():
        print()
        print(_board_ascii(board, human))
        if board.turn == human:
            try:
                text = input('Your move: ').strip()
            except EOFError:
                print()
                return board
            if text.lower() in ('quit', 'exit', 'q'):
                return board
            move = None
            try:
                move = chess.Move.from_uci(text)
                if move not in board.legal_moves:
                    move = None
            except ValueError:
                pass
            if move is None:
                try:
                    move = board.parse_san(text)
                except ValueError:
                    print('Illegal move.')
                    continue
            board.push(move)
        else:
            move = _model_move(model, board, temperature)
            _, value = model.evaluate_board(board)
            print(f'TopCat plays: {board.san(move)} ({move.uci()}) | value {value:+.3f}')
            board.push(move)
    print()
    print(_board_ascii(board, human))
    print(f'Game over: {board.result()}')
    return board


def run_test(model):
    """Forward pass on the initial position and on a batch of 8, best move and KR vs K analysis."""
    model.eval()
    device = model.device
    tokenizer = model.tokenizer

    board = chess.Board()
    tokens = [t.unsqueeze(0).to(device) for t in tokenizer.tokenize_board(board)]
    with torch.no_grad():
        logits, value = model(*tokens)
    print(f'Initial position: logits {tuple(logits.shape)}, value {tuple(value.shape)} = {value.item():+.4f}')
    print(f'Best move (initial position): {board.san(model.best_legal_move(board))}')

    rng = random.Random(0)
    boards = []
    b = chess.Board()
    for _ in range(8):
        boards.append(b.copy())
        for _ in range(rng.randint(1, 6)):
            if b.is_game_over():
                b = chess.Board()
            b.push(rng.choice(list(b.legal_moves)))
    batch = [t.to(device) for t in tokenizer.tokenize_batch(boards)]
    with torch.no_grad():
        logits, value = model(*batch)
    print(f'Batch of 8: logits {tuple(logits.shape)}, value {tuple(value.shape)}')
    print()
    analyze_position(model, '4k3/8/8/8/8/8/8/4KR2 w - - 0 1')


# =============================================================================
# 7. CLI
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description='TopCat chess engine (transformer + Neural ODE)')
    parser.add_argument('--mode', default='test',
                        choices=['eval_train', 'selfplay', 'test', 'play', 'hpo',
                                 'policy_data', 'policy_train', 'vs_random', 'pretrain_rules'])
    parser.add_argument('--load', default=None, help='checkpoint (state_dict) to load')
    parser.add_argument('--embed', type=int, default=256)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--color', default='white', choices=['white', 'black'], help='human color in play mode')
    parser.add_argument('--pgn', default=None, help="policy_data: .pgn/.pgn.zst ('-' = stdin)")
    parser.add_argument('--data', default=None, help='training JSON')
    parser.add_argument('--out', default=None, help='output file (checkpoint or JSON)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--games', type=int, default=None, help='games per color vs random')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--curriculum', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--random-positions', type=int, default=0,
                        help='pretrain_rules: number of random-game positions (added to --data)')
    parser.add_argument('--tasks', default=','.join(DEFAULT_PRETRAIN_TASKS),
                        help=f'pretrain_rules: comma-separated tasks ({", ".join(PRETRAIN_TASKS)})')
    parser.add_argument('--workers', type=int, default=0, help='pretrain_rules: DataLoader workers')
    parser.add_argument('--resume', action='store_true',
                        help='policy_train / pretrain_rules / eval_train: continue from <out>_state.pt '
                             '(written after every epoch) if it exists')
    args = parser.parse_args(argv)

    def pick(value, default):
        return default if value is None else value

    if args.mode == 'hpo':
        try:
            run_hpo()
        except NotImplementedError as e:
            print(e)
            return 2

    if args.mode == 'policy_data':
        if args.pgn is None:
            parser.error('--mode policy_data requires --pgn')
        build_policy_dataset_from_pgn(args.pgn, pick(args.out, 'policy_dataset.json'),
                                      max_positions=pick(args.max_samples, 500_000))
        return 0

    model_kwargs = {'embed_dim': args.embed, 'num_heads': args.heads, 'n_layers': args.layers}
    model = load_model(args.load, **model_kwargs) if args.load else build_model(**model_kwargs)

    if args.mode == 'test':
        run_test(model)
    elif args.mode == 'play':
        play_interactive(model, human_color=args.color, temperature=pick(args.temperature, 0.0))
    elif args.mode == 'vs_random':
        evaluate_vs_random(model, n_games=pick(args.games, 40), temperature=pick(args.temperature, 0.5))
    elif args.mode == 'eval_train':
        train_evaluation(model, json_path=pick(args.data, 'scores_dataset/ds1.json'),
                         n_epochs=pick(args.epochs, 20), batch_size=pick(args.batch_size, 256),
                         lr=pick(args.lr, 1e-4), max_samples=args.max_samples,
                         checkpoint_path=pick(args.out, 'topcat_eval.pt'), resume=args.resume)
    elif args.mode == 'policy_train':
        train_policy_supervised(model, json_path=pick(args.data, 'policy_dataset.json'),
                                n_epochs=pick(args.epochs, 50), batch_size=pick(args.batch_size, 1024),
                                lr=pick(args.lr, 2e-4), max_samples=args.max_samples,
                                checkpoint_path=pick(args.out, 'topcat_policy.pt'), resume=args.resume)
        evaluate_vs_random(model, n_games=pick(args.games, 40), temperature=pick(args.temperature, 0.5))
    elif args.mode == 'selfplay':
        temperatures = (0.8, 1.2) if args.temperature is None else (args.temperature, args.temperature)
        run_selfplay_symmetric_loop(model, n_episodes=args.episodes, lr=pick(args.lr, 1e-5),
                                    temperatures=temperatures, curriculum_level=args.curriculum,
                                    eval_games=pick(args.games, 10),
                                    checkpoint_path=pick(args.out, 'topcat_selfplay.pt'))
    elif args.mode == 'pretrain_rules':
        tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
        unknown = [t for t in tasks if t not in PRETRAIN_TASKS]
        if unknown:
            parser.error(f'unknown tasks: {unknown}')
        fens = load_fens(args.data) if args.data else []
        if args.max_samples is not None:
            fens = fens[:args.max_samples]
        if args.random_positions:
            # fixed seed: the same position list (and validation split) when resuming
            fens += generate_random_fens(args.random_positions, seed=0)
        if not fens:
            parser.error('--mode pretrain_rules requires --data and/or --random-positions')
        out = pick(args.out, 'topcat_pretrain.pt')
        heads_path = (out[:-3] if out.endswith('.pt') else out) + '_heads.pt'
        pretrain_rules(model, fens, tasks=tasks, n_epochs=pick(args.epochs, 5),
                       batch_size=pick(args.batch_size, 512), lr=pick(args.lr, 3e-4),
                       num_workers=args.workers, checkpoint_path=out, heads_path=heads_path,
                       resume=args.resume)
    return 0


if __name__ == '__main__':
    sys.exit(main())
