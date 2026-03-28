# hchessai_uci_kednaik_vit.py
import sys
import traceback
from pathlib import Path
from typing import Optional

import chess
import torch
import torch.nn as nn

ENGINE_NAME = "hchessai"
ENGINE_AUTHOR = "Dirk D. Sommerfeld / K Naik"

_ALLOWED_STDOUT_PREFIXES = (
    "id ",
    "option ",
    "uciok",
    "readyok",
    "bestmove ",
    "info ",
)

class UCIStdoutFilter:
    def __init__(self, real_stdout):
        self.real_stdout = real_stdout

    def write(self, data: str):
        lines = data.splitlines(keepends=True)
        for line in lines:
            text = line.strip()
            if text == "" or text.startswith(_ALLOWED_STDOUT_PREFIXES):
                self.real_stdout.write(line)
            else:
                sys.stderr.write(line)

    def flush(self):
        self.real_stdout.flush()

sys.stdout = UCIStdoutFilter(sys.stdout)

def send(msg: str) -> None:
    print(msg, flush=True)

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

VOCAB_VIT = {
    'EMPTY': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
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
    '[PAD]': 43, '[MASK]': 44
}

class ViTChessHybrid(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = len(VOCAB_VIT)
        self.meta_embedding = nn.Embedding(self.vocab_size, d_model)
        self.meta_pos_encoding = nn.Parameter(torch.randn(1, 9, d_model))
        self.board_embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed_2d = nn.Parameter(torch.randn(1, 64, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, self.vocab_size)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 73, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
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
        value_score = self.value_head(flat_latent)
        return policy_logits, value_score

def encode_state_vit(board: chess.Board, win_token: int, white_elo_str: str, black_elo_str: str):
    def get_elo_token(elo_str):
        try:
            elo = int(elo_str)
            if elo < 1000:
                return VOCAB_VIT['[ELO_0]']
            if elo >= 2600:
                return VOCAB_VIT['[ELO_9]']
            return VOCAB_VIT[f'[ELO_{(elo - 1000) // 200 + 1}]']
        except Exception:
            return VOCAB_VIT['[ELO_4]']

    w_elo = get_elo_token(white_elo_str)
    b_elo = get_elo_token(black_elo_str)
    seq = [win_token, w_elo, b_elo]

    if board.fullmove_number == 1 and board.turn == chess.WHITE:
        seq.append(VOCAB_VIT['[INIT]'])
    else:
        seq.append(VOCAB_VIT['[W_MOVE]'] if board.turn == chess.WHITE else VOCAB_VIT['[B_MOVE]'])

    NC = VOCAB_VIT['[NO_CASTLE]']
    seq.append(VOCAB_VIT['[W_KSC]'] if board.has_kingside_castling_rights(chess.WHITE) else NC)
    seq.append(VOCAB_VIT['[W_QSC]'] if board.has_queenside_castling_rights(chess.WHITE) else NC)
    seq.append(VOCAB_VIT['[B_KSC]'] if board.has_kingside_castling_rights(chess.BLACK) else NC)
    seq.append(VOCAB_VIT['[B_QSC]'] if board.has_queenside_castling_rights(chess.BLACK) else NC)

    ep = board.ep_square
    seq.append(VOCAB_VIT[f'[EP_{chess.square_name(ep)[0].upper()}]'] if ep is not None else VOCAB_VIT['[EP_NONE]'])

    for s in chess.SQUARES:
        p = board.piece_at(s)
        seq.append(VOCAB_VIT[p.symbol()] if p else VOCAB_VIT['EMPTY'])

    return seq

def get_best_move_vit(model: ViTChessHybrid, board: chess.Board, device: torch.device, whiteElo: str, blackElo: str):
    model.eval()
    win_token = VOCAB_VIT['[W_WIN]'] if board.turn == chess.WHITE else VOCAB_VIT['[B_WIN]']
    input_seq = encode_state_vit(board, win_token, whiteElo, blackElo)
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, _ = model(input_tensor)
        probs = torch.softmax(policy_logits[0], dim=-1)

    best_move = None
    best_score = -float('inf')
    for move in board.legal_moves:
        board.push(move)
        try:
            score = 0.0
            for s in chess.SQUARES:
                p = board.piece_at(s)
                target_token = VOCAB_VIT[p.symbol()] if p else VOCAB_VIT['EMPTY']
                score += torch.log(probs[s, target_token] + 1e-9).item()
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

class ModelRuntime:
    def __init__(self):
        self.device: Optional[torch.device] = None
        self.model: Optional[ViTChessHybrid] = None
        self.loaded = False
        self.error: Optional[str] = None
        self.model_path = app_dir() / "checkpoints" / "chess_vit_epoch1_step8000.pt"

    def set_model_path(self, raw_value: str) -> None:
        p = Path(raw_value)
        if not p.is_absolute():
            p = (app_dir() / p).resolve()
        self.model_path = p
        self.loaded = False
        self.model = None
        self.error = None

    def ensure_loaded(self) -> None:
        if self.loaded:
            return
        try:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            model = ViTChessHybrid(d_model=256, nhead=8, num_layers=8, dropout=0.1).to(self.device)
            try:
                ckpt = torch.load(str(self.model_path), map_location=self.device, weights_only=True)
            except TypeError:
                ckpt = torch.load(str(self.model_path), map_location=self.device)

            if isinstance(ckpt, dict) and 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            else:
                model.load_state_dict(ckpt)

            model.eval()
            self.model = model
            self.loaded = True
            self.error = None
            log(f"Loaded model from {self.model_path} on {self.device}")
        except Exception as exc:
            self.loaded = False
            self.model = None
            self.error = f"{type(exc).__name__}: {exc}"
            log(f"Model load failed: {self.error}")

class EngineState:
    def __init__(self):
        self.board = chess.Board()
        self.runtime = ModelRuntime()
        self.uci_elo = 2600
        self.debug = False

STATE = EngineState()

def handle_uci() -> None:
    send(f"id name {ENGINE_NAME}")
    send(f"id author {ENGINE_AUTHOR}")
    send("option name UCI_Elo type spin default 2600 min 900 max 2600")
    send("option name ModelPath type string default checkpoints/chess_vit_epoch1_step8000.pt")
    send("option name DebugLog type check default false")
    send("uciok")

def handle_setoption(cmd: str) -> None:
    lower = cmd.lower()
    if "name uci_elo value " in lower:
        raw = lower.split("name uci_elo value ", 1)[1].strip()
        try:
            STATE.uci_elo = max(900, min(2600, int(raw)))
        except Exception:
            pass
        return
    if "name modelpath value " in lower:
        raw = cmd.split("value", 1)[1].strip()
        STATE.runtime.set_model_path(raw)
        return
    if "name debuglog value " in lower:
        raw = lower.split("name debuglog value ", 1)[1].strip()
        STATE.debug = (raw == "true")
        return

def handle_isready() -> None:
    STATE.runtime.ensure_loaded()
    send("readyok")

def set_position(cmd: str) -> None:
    parts = cmd.split()
    if len(parts) < 2:
        STATE.board = chess.Board()
        return

    if parts[1] == "startpos":
        board = chess.Board()
        if "moves" in parts:
            idx = parts.index("moves")
            for mv in parts[idx + 1:]:
                try:
                    board.push_uci(mv)
                except Exception:
                    break
        STATE.board = board
        return

    if parts[1] == "fen":
        fen_parts = []
        i = 2
        while i < len(parts) and parts[i] != "moves":
            fen_parts.append(parts[i])
            i += 1
        try:
            board = chess.Board(" ".join(fen_parts))
        except Exception:
            board = chess.Board()
        if i < len(parts) and parts[i] == "moves":
            for mv in parts[i + 1:]:
                try:
                    board.push_uci(mv)
                except Exception:
                    break
        STATE.board = board

def handle_go() -> None:
    try:
        STATE.runtime.ensure_loaded()
        if not STATE.runtime.loaded or STATE.runtime.model is None or STATE.runtime.device is None:
            send("info string model not loaded")
            send("bestmove 0000")
            return

        elo_str = str(STATE.uci_elo)
        best_move = get_best_move_vit(
            STATE.runtime.model,
            STATE.board,
            STATE.runtime.device,
            elo_str,
            elo_str,
        )
        if best_move is None:
            send("info string no move generated")
            send("bestmove 0000")
            return

        send(f"info string source=vit_hybrid uci_elo={STATE.uci_elo}")
        send(f"bestmove {best_move.uci()}")
    except Exception as exc:
        log(f"go failed: {exc}")
        if STATE.debug:
            log(traceback.format_exc())
        send("bestmove 0000")

def main() -> None:
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        cmd = line.strip()
        if not cmd:
            continue
        try:
            if cmd == "uci":
                handle_uci()
            elif cmd == "isready":
                handle_isready()
            elif cmd == "ucinewgame":
                STATE.board = chess.Board()
            elif cmd.startswith("setoption"):
                handle_setoption(cmd)
            elif cmd.startswith("position"):
                set_position(cmd)
            elif cmd.startswith("go"):
                handle_go()
            elif cmd == "stop":
                pass
            elif cmd == "debug on":
                STATE.debug = True
            elif cmd == "debug off":
                STATE.debug = False
            elif cmd == "quit":
                break
        except Exception as exc:
            log(f"main loop error: {exc}")
            if STATE.debug:
                log(traceback.format_exc())
            send("bestmove 0000")

if __name__ == "__main__":
    main()
