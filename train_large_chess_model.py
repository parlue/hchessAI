#!/usr/bin/env python3
"""
Train Large Chess Model from ChessData PGNs
Author: Dirk D. Sommerfeld
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, List, Tuple

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

VOCAB_VIT = {
    'EMPTY': 0,
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
    '[W_WIN]': 13, '[B_WIN]': 14, '[DRAW]': 15, '[INIT]': 16,
    '[W_MOVE]': 17, '[B_MOVE]': 18,
    '[W_KSC]': 19, '[W_QSC]': 20, '[B_KSC]': 21, '[B_QSC]': 22,
    '[NO_CASTLE]': 23,
    '[EP_A]': 24, '[EP_B]': 25, '[EP_C]': 26, '[EP_D]': 27,
    '[EP_E]': 28, '[EP_F]': 29, '[EP_G]': 30, '[EP_H]': 31,
    '[EP_NONE]': 32,
    '[ELO_0]': 33, '[ELO_1]': 34, '[ELO_2]': 35, '[ELO_3]': 36,
    '[ELO_4]': 37, '[ELO_5]': 38, '[ELO_6]': 39, '[ELO_7]': 40,
    '[ELO_8]': 41, '[ELO_9]': 42,
    '[PAD]': 43, '[MASK]': 44,
}

def get_elo_token(elo_str: str) -> int:
    try:
        elo = int(elo_str)
        if elo < 1000:
            return VOCAB_VIT['[ELO_0]']
        if elo >= 2600:
            return VOCAB_VIT['[ELO_9]']
        return VOCAB_VIT[f'[ELO_{(elo - 1000) // 200 + 1}]']
    except Exception:
        return VOCAB_VIT['[ELO_4]']

def encode_state_vit(board: chess.Board, win_token: int, white_elo_str: str, black_elo_str: str) -> List[int]:
    w_elo = get_elo_token(white_elo_str)
    b_elo = get_elo_token(black_elo_str)
    seq = [win_token, w_elo, b_elo]
    if board.fullmove_number == 1 and board.turn == chess.WHITE:
        seq.append(VOCAB_VIT['[INIT]'])
    else:
        seq.append(VOCAB_VIT['[W_MOVE]'] if board.turn == chess.WHITE else VOCAB_VIT['[B_MOVE]'])
    nc = VOCAB_VIT['[NO_CASTLE]']
    seq.append(VOCAB_VIT['[W_KSC]'] if board.has_kingside_castling_rights(chess.WHITE) else nc)
    seq.append(VOCAB_VIT['[W_QSC]'] if board.has_queenside_castling_rights(chess.WHITE) else nc)
    seq.append(VOCAB_VIT['[B_KSC]'] if board.has_kingside_castling_rights(chess.BLACK) else nc)
    seq.append(VOCAB_VIT['[B_QSC]'] if board.has_queenside_castling_rights(chess.BLACK) else nc)
    ep = board.ep_square
    seq.append(VOCAB_VIT[f'[EP_{chess.square_name(ep)[0].upper()}]'] if ep else VOCAB_VIT['[EP_NONE]'])
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        seq.append(VOCAB_VIT[p.symbol()] if p else VOCAB_VIT['EMPTY'])
    return seq

def encode_board_after_move(board_after: chess.Board) -> List[int]:
    out = []
    for sq in chess.SQUARES:
        p = board_after.piece_at(sq)
        out.append(VOCAB_VIT[p.symbol()] if p else VOCAB_VIT['EMPTY'])
    return out

def result_value_for_side_to_move(result: str, side_to_move: chess.Color) -> float:
    if result == "1-0":
        white_value = 1.0
    elif result == "0-1":
        white_value = -1.0
    else:
        white_value = 0.0
    return white_value if side_to_move == chess.WHITE else -white_value

class ViTChessHybrid(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 8, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = len(VOCAB_VIT)
        self.meta_embedding = nn.Embedding(self.vocab_size, d_model)
        self.meta_pos_encoding = nn.Parameter(torch.randn(1, 9, d_model))
        self.board_embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed_2d = nn.Parameter(torch.randn(1, 64, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, self.vocab_size)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 73, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        meta_tokens = x[:, :9]
        meta_emb = self.meta_embedding(meta_tokens) + self.meta_pos_encoding
        board_tokens = x[:, 9:]
        board_emb = self.board_embedding(board_tokens) + self.pos_embed_2d
        full_seq = torch.cat([meta_emb, board_emb], dim=1)
        latent = self.transformer(full_seq)
        board_latent = latent[:, 9:, :]
        policy_logits = self.policy_head(board_latent)
        flat_latent = latent.reshape(batch_size, -1)
        value_score = self.value_head(flat_latent).squeeze(-1)
        return policy_logits, value_score

class PGNIterableDataset(IterableDataset):
    def __init__(self, pgn_files: List[Path], max_games: int | None = None, max_plies_per_game: int | None = None):
        super().__init__()
        self.pgn_files = pgn_files
        self.max_games = max_games
        self.max_plies_per_game = max_plies_per_game

    def __iter__(self):
        yielded_games = 0
        for pgn_path in self.pgn_files:
            with pgn_path.open("r", encoding="utf-8", errors="ignore") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    result = game.headers.get("Result", "*")
                    white_elo = game.headers.get("WhiteElo", "2600")
                    black_elo = game.headers.get("BlackElo", "2600")
                    if result not in ("1-0", "0-1", "1/2-1/2"):
                        continue
                    board = game.board()
                    plies = 0
                    for move in game.mainline_moves():
                        if self.max_plies_per_game is not None and plies >= self.max_plies_per_game:
                            break
                        side_to_move = board.turn
                        win_token = VOCAB_VIT['[W_WIN]'] if side_to_move == chess.WHITE else VOCAB_VIT['[B_WIN]']
                        input_seq = encode_state_vit(board, win_token, white_elo, black_elo)
                        board.push(move)
                        target_board = encode_board_after_move(board)
                        value_target = result_value_for_side_to_move(result, side_to_move)
                        yield input_seq, target_board, value_target
                        plies += 1
                    yielded_games += 1
                    if self.max_games is not None and yielded_games >= self.max_games:
                        return

def collate_fn(batch):
    xs, ys, vs = zip(*batch)
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    v = torch.tensor(vs, dtype=torch.float32)
    return x, y, v

def find_pgns(chessdata_dir: Path) -> List[Path]:
    files = sorted(chessdata_dir.rglob("*.pgn"))
    if not files:
        raise FileNotFoundError(f"No .pgn files found under {chessdata_dir}")
    return files

def save_checkpoint(model, optimizer, epoch, step, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step},
        out_path,
    )

def train(args):
    base_dir = Path(args.base_dir).resolve()
    chessdata_dir = base_dir / "ChessData"
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    pgn_files = find_pgns(chessdata_dir)
    print("Using PGN files:")
    for p in pgn_files:
        print(" -", p)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    dataset = PGNIterableDataset(
        pgn_files=pgn_files,
        max_games=args.max_games,
        max_plies_per_game=args.max_plies_per_game,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn)

    model = ViTChessHybrid(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_policy = 0.0
        running_value = 0.0
        running_total = 0.0
        batch_count = 0

        for x, y, v in pbar:
            x = x.to(device)
            y = y.to(device)
            v = v.to(device)

            policy_logits, value_pred = model(x)
            policy_loss = F.cross_entropy(policy_logits.reshape(-1, policy_logits.size(-1)), y.reshape(-1))
            value_loss = F.mse_loss(value_pred, v)
            loss = policy_loss + args.value_loss_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            batch_count += 1
            running_policy += float(policy_loss.item())
            running_value += float(value_loss.item())
            running_total += float(loss.item())

            pbar.set_postfix(
                policy=f"{running_policy / batch_count:.4f}",
                value=f"{running_value / batch_count:.4f}",
                total=f"{running_total / batch_count:.4f}",
            )

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                ckpt_path = checkpoints_dir / f"chess_vit_epoch{epoch}_step{global_step}.pt"
                save_checkpoint(model, optimizer, epoch, global_step, ckpt_path)

        latest_path = checkpoints_dir / "chess_vit_latest.pt"
        epoch_path = checkpoints_dir / f"chess_vit_epoch{epoch}.pt"
        save_checkpoint(model, optimizer, epoch, global_step, latest_path)
        save_checkpoint(model, optimizer, epoch, global_step, epoch_path)

    print("Training finished.")
    print("Latest checkpoint:", checkpoints_dir / "chess_vit_latest.pt")

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", help="Project root containing ChessData/ and checkpoints/")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--value-loss-weight", type=float, default=0.25)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--max-plies-per-game", type=int, default=200)
    ap.add_argument("--save-every-steps", type=int, default=1000)
    ap.add_argument("--cpu", action="store_true")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
