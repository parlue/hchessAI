# hchessai_full_uci_level.py
# UCI engine shell for hchessai
# - BearChess-friendly startup
# - lazy checkpoint loading
# - UCI SkillLevel 0..100
# - default checkpoint: checkpoints/chess_vit_epoch1_step4000.pt
# - robust fallback move selection with python-chess
#
# Notes:
# This file is wired as if the engine were finished from the UCI side.
# The actual model inference hook is concentrated in one method:
#     HChessModel.pick_move(...)
# If your network forward signature differs, adapt only that method.

import os
import sys
import time
import random
import traceback
from pathlib import Path
from typing import Optional, List, Tuple

try:
    import chess
except Exception as exc:
    print(f"python-chess import failed: {exc}", file=sys.stderr, flush=True)
    raise

ENGINE_NAME = "hchessai"
ENGINE_AUTHOR = "Dirk D. Sommerfeld"

ALLOWED_STDOUT_PREFIXES = (
    "id ",
    "option ",
    "uciok",
    "readyok",
    "bestmove ",
    "info ",
)

# ----------------------------
# Stdout filter
# ----------------------------

class UCIStdoutFilter:
    def __init__(self, real_stdout):
        self.real_stdout = real_stdout

    def write(self, data: str):
        lines = data.splitlines(keepends=True)
        for line in lines:
            text = line.strip()
            if text == "" or text.startswith(ALLOWED_STDOUT_PREFIXES):
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


# ----------------------------
# Model wrapper
# ----------------------------

class HChessModel:
    def __init__(self):
        self.torch = None
        self.state = None
        self.loaded = False
        self.error: Optional[str] = None
        self.checkpoint_path: Optional[Path] = None

    def load(self, checkpoint_path: Path) -> None:
        if self.loaded:
            return
        try:
            import torch  # lazy import
            self.torch = torch
            self.checkpoint_path = checkpoint_path
            self.state = torch.load(str(checkpoint_path), map_location="cpu")
            self.loaded = True
            self.error = None
            log(f"checkpoint loaded: {checkpoint_path}")
        except Exception as exc:
            self.loaded = False
            self.error = f"{type(exc).__name__}: {exc}"
            log(f"checkpoint load failed: {self.error}")

    def has_loaded_checkpoint(self) -> bool:
        return self.loaded

    def pick_move(self, board: "chess.Board", skill_level: int) -> Optional[str]:
        """
        Optional model inference hook.
        Return a UCI move string or None.

        This method is intentionally conservative:
        - It only tries model inference if a checkpoint was loaded.
        - Because architecture / tensor encoding can differ between projects,
          it safely returns None by default unless you adapt it.

        To integrate your actual model, replace the body below with:
        1) board -> tensor encoding
        2) forward pass
        3) legal-move masking
        4) move selection
        """
        if not self.loaded:
            return None

        # Safe default for now. UCI layer remains fully functional.
        return None


# ----------------------------
# UCI state
# ----------------------------

class EngineState:
    def __init__(self):
        self.board = chess.Board()
        self.model = HChessModel()
        self.model_path = self.default_model_path()
        self.skill_level = 100
        self.use_model = True
        self.debug_log = False
        self.rand = random.Random(42)
        self.last_go_started_at = 0.0

    def default_model_path(self) -> Path:
        base = app_dir()
        return base / "checkpoints" / "chess_vit_epoch1_step4000.pt"

    def set_model_path(self, value: str) -> None:
        p = Path(value)
        if not p.is_absolute():
            p = (app_dir() / p).resolve()
        self.model_path = p

    def ensure_model_loaded(self) -> None:
        if not self.use_model:
            return
        if self.model.has_loaded_checkpoint():
            return
        self.model.load(self.model_path)

STATE = EngineState()


# ----------------------------
# Position handling
# ----------------------------

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
        fen = " ".join(fen_parts)
        try:
            board = chess.Board(fen)
        except Exception:
            board = chess.Board()
        if i < len(parts) and parts[i] == "moves":
            for mv in parts[i + 1:]:
                try:
                    board.push_uci(mv)
                except Exception:
                    break
        STATE.board = board
        return


# ----------------------------
# Move scoring / level system
# ----------------------------

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def move_score(board: "chess.Board", move: "chess.Move") -> int:
    score = 0

    # Promotion
    if move.promotion:
        score += PIECE_VALUE.get(move.promotion, 0) + 400

    # Capture value
    if board.is_capture(move):
        if board.is_en_passant(move):
            score += PIECE_VALUE[chess.PAWN] + 60
        else:
            captured = board.piece_at(move.to_square)
            if captured is not None:
                score += PIECE_VALUE.get(captured.piece_type, 0) + 60

        mover = board.piece_at(move.from_square)
        captured = board.piece_at(move.to_square)
        if mover and captured:
            score += max(0, PIECE_VALUE[captured.piece_type] - PIECE_VALUE[mover.piece_type] // 4)

    # Check / mate
    board.push(move)
    if board.is_checkmate():
        score += 100000
    elif board.is_check():
        score += 250

    # Mobility / center / development
    to_sq = move.to_square
    if to_sq in (chess.D4, chess.E4, chess.D5, chess.E5):
        score += 35
    elif chess.square_file(to_sq) in (2, 3, 4, 5) and chess.square_rank(to_sq) in (2, 3, 4, 5):
        score += 12

    if board.fullmove_number <= 10:
        moved_piece = board.piece_at(to_sq)
        if moved_piece and moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            score += 18

    # Avoid stalemate-ish nonsense slightly
    if board.is_stalemate():
        score -= 500

    board.pop()
    return score


def choose_move_by_level(board: "chess.Board", skill_level: int, rng: random.Random) -> str:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "0000"

    scored: List[Tuple[int, chess.Move]] = [(move_score(board, mv), mv) for mv in legal_moves]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Skill 100: almost best heuristic move
    # Skill 0: much wider randomness / occasional blunders
    level = max(0, min(100, skill_level))

    # Forced mate or immediate mate defense still gets priority
    if scored and scored[0][0] >= 100000:
        return scored[0][1].uci()

    # Compute top-k window from level
    # 100 -> top 1
    # 75 -> top 3
    # 50 -> top ~6
    # 25 -> top ~12
    # 0  -> all legal moves
    n = len(scored)
    if level >= 98:
        window = 1
    else:
        frac = 1.0 - (level / 100.0)
        window = 1 + int(frac * max(1, min(20, n - 1)))
        window = max(1, min(n, window))

    candidates = scored[:window]

    # Additional blunder chance on low levels
    blunder_prob = max(0.0, (45 - level) / 100.0) if level < 45 else 0.0
    if rng.random() < blunder_prob and n > 1:
        tail_start = min(max(window, n // 2), n - 1)
        return rng.choice(scored[tail_start:])[1].uci()

    # Weighted pick from candidate window
    weights = []
    base = candidates[0][0]
    temp = 0.15 + (1.85 * (1.0 - level / 100.0))  # higher temperature on lower levels

    for score, _mv in candidates:
        # stable positive weight
        delta = max(-2000, min(2000, score - base))
        weight = pow(2.718281828, delta / (250.0 * temp))
        weights.append(max(1e-6, weight))

    chosen = rng.choices(candidates, weights=weights, k=1)[0][1]
    return chosen.uci()


# ----------------------------
# Time handling
# ----------------------------

def parse_go_limits(cmd: str) -> dict:
    parts = cmd.split()
    limits = {}
    i = 1
    while i < len(parts):
        token = parts[i]
        if token in {"wtime", "btime", "winc", "binc", "movetime", "movestogo", "depth", "nodes", "mate"}:
            if i + 1 < len(parts):
                try:
                    limits[token] = int(parts[i + 1])
                except Exception:
                    pass
                i += 2
                continue
        elif token in {"infinite", "ponder"}:
            limits[token] = True
        i += 1
    return limits


# ----------------------------
# UCI handlers
# ----------------------------

def handle_uci() -> None:
    send(f"id name {ENGINE_NAME}")
    send(f"id author {ENGINE_AUTHOR}")
    send(f"option name SkillLevel type spin default 100 min 0 max 100")
    send(f"option name ModelPath type string default checkpoints/chess_vit_epoch1_step4000.pt")
    send(f"option name UseModel type check default true")
    send(f"option name DebugLog type check default false")
    send("uciok")


def handle_setoption(cmd: str) -> None:
    lower = cmd.lower()
    try:
        if "name skilllevel value " in lower:
            raw = lower.split("name skilllevel value ", 1)[1].strip()
            STATE.skill_level = max(0, min(100, int(raw)))
            return

        if "name modelpath value " in lower:
            raw = cmd.split("value", 1)[1].strip()
            STATE.set_model_path(raw)
            # allow reload on next isready if path changes
            STATE.model = HChessModel()
            return

        if "name usemodel value " in lower:
            raw = lower.split("name usemodel value ", 1)[1].strip()
            STATE.use_model = (raw == "true")
            return

        if "name debuglog value " in lower:
            raw = lower.split("name debuglog value ", 1)[1].strip()
            STATE.debug_log = (raw == "true")
            return
    except Exception as exc:
        log(f"setoption failed: {exc}")


def handle_isready() -> None:
    STATE.ensure_model_loaded()
    send("readyok")


def handle_go(cmd: str) -> None:
    STATE.last_go_started_at = time.time()
    limits = parse_go_limits(cmd)

    # Optional model move
    model_move = None
    if STATE.use_model:
        STATE.ensure_model_loaded()
        try:
            model_move = STATE.model.pick_move(STATE.board, STATE.skill_level)
        except Exception as exc:
            log(f"model pick failed: {exc}")
            if STATE.debug_log:
                log(traceback.format_exc())

    if model_move:
        send(f"info string source=model skill={STATE.skill_level}")
        send(f"bestmove {model_move}")
        return

    # Fallback move
    move = choose_move_by_level(STATE.board, STATE.skill_level, STATE.rand)
    send(f"info string source=fallback skill={STATE.skill_level}")
    send(f"bestmove {move}")


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
                handle_go(cmd)

            elif cmd == "stop":
                pass

            elif cmd == "quit":
                break

            elif cmd == "debug on":
                STATE.debug_log = True

            elif cmd == "debug off":
                STATE.debug_log = False

        except Exception as exc:
            log(f"main loop error: {exc}")
            if STATE.debug_log:
                log(traceback.format_exc())
            # Never hang a GUI; always remain responsive
            send("bestmove 0000")


if __name__ == "__main__":
    main()
