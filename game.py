from __future__ import annotations

import re
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage

WINNING_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


class GameState:
    def __init__(self) -> None:
        self.board: list[int] = [-1] * 9
        self.current_player: int = 1

    def is_valid_move(self, position: int) -> bool:
        if position < 0 or position > 8:
            return False
        return self.board[position] == -1

    def make_move(self, position: int, player: int) -> None:
        if not self.is_valid_move(position):
            raise ValueError(
                f"Invalid move: position {position} is out of range or already occupied"
            )
        if player not in (0, 1):
            raise ValueError(f"Invalid player: {player}. Must be 0 (O) or 1 (X)")
        if player != self.current_player:
            raise ValueError(
                f"Not your turn. Current player: {self.current_player}"
            )
        self.board[position] = player
        self.current_player = 1 - player

    def check_winner(self) -> int | None:
        for a, b, c in WINNING_LINES:
            if (
                self.board[a] != -1
                and self.board[a] == self.board[b] == self.board[c]
            ):
                return self.board[a]
        return None

    def is_draw(self) -> bool:
        return -1 not in self.board and self.check_winner() is None

    def is_game_over(self) -> bool:
        return self.check_winner() is not None or self.is_draw()

    def display(self) -> str:
        symbols = {-1: ".", 0: "O", 1: "X"}
        rows = []
        for i in range(3):
            row_start = i * 3
            row = " | ".join(
                f" {symbols[self.board[row_start + j]]} "
                for j in range(3)
            )
            rows.append(row)
        sep = "---+---+---"
        return "\n" + f"\n{sep}\n".join(rows) + "\n"

    def board_with_positions(self) -> str:
        symbols = {-1: ".", 0: "O", 1: "X"}
        rows = []
        for i in range(3):
            row_start = i * 3
            cells = []
            for j in range(3):
                pos = row_start + j
                val = self.board[pos]
                cell = f"{symbols[val]}({pos})" if val != -1 else f".({pos})"
                cells.append(cell)
            rows.append(" | ".join(cells))
        sep = "---+---+---"
        return "\n" + f"\n{sep}\n".join(rows) + "\n"

    def __repr__(self) -> str:
        return f"GameState(board={self.board}, current_player={self.current_player})"


def human_player(game_state: GameState, identity: int) -> int:
    symbol = "X" if identity == 1 else "O"
    print(f"\nYour turn! You are {symbol}")
    print(game_state.display())
    print("Positions: 0|1|2 / 3|4|5 / 6|7|8")

    while True:
        try:
            pos = int(input("Enter position (0-8): "))
            if game_state.is_valid_move(pos):
                return pos
            print(f"Position {pos} is not available. Try again.")
        except ValueError:
            print("Please enter a number between 0 and 8.")
        except KeyboardInterrupt:
            print("\nGame cancelled.")
            raise


def _simulate_stream(text: str, delay: float = 0.01) -> None:
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
        sys.stdout.flush()


def _clean_reasoning(text: str) -> str:
    text = re.sub(r'functions?\.make_move\s*\(\s*\{[^}]*\}\s*\)', '', text)
    text = re.sub(r'make_move\s*\(\s*\{[^}]*\}\s*\)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_position_from_text(text: str) -> int | None:
    patterns = [
        r'position["\s]*[:=]\s*(\d)',
        r'position\s*=\s*(\d)',
        r'at\s+position\s+(\d)',
        r'play\s+(?:position\s+)?(\d)',
        r'place\s+(?:at\s+)?(?:position\s+)?(\d)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pos = int(match.group(1))
            if 0 <= pos <= 8:
                return pos
    return None


def agent_player_fn(llm_with_tools, system_prompt: str, game_state: GameState, identity: int) -> int:
    symbol = "X" if identity == 1 else "O"
    prompt = (
        f"Current board state: {game_state.board}\n"
        f"Board layout with positions:\n{game_state.board_with_positions()}\n"
        f"You are {symbol} (value={identity}). Make your move."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    response = llm_with_tools.invoke(messages)

    reasoning = response.content if isinstance(response.content, str) else ""
    if isinstance(response.content, list):
        parts = []
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        reasoning = "".join(parts)

    clean = _clean_reasoning(reasoning)
    if clean:
        _simulate_stream(clean)
        print("─────────────────")

    position = None

    if response.tool_calls:
        position = response.tool_calls[0]["args"]["position"]
        if not isinstance(position, int):
            position = int(position)

    if position is None:
        position = _extract_position_from_text(reasoning)

    if position is None:
        raise ValueError(
            f"Agent did not make a valid move. Response: {reasoning}"
        )

    return position


def play_game(
    player1_fn,
    player2_fn,
    game_state: GameState | None = None,
) -> dict:
    if game_state is None:
        game_state = GameState()

    print("\n=== JIRO KATA ===")
    print("X (1) goes first!")
    print(game_state.display())

    move_count = 0
    while not game_state.is_game_over():
        current_identity = 1 if move_count % 2 == 0 else 0
        player_fn = player1_fn if current_identity == 1 else player2_fn
        symbol = "X" if current_identity == 1 else "O"

        print(f"\n--- {symbol}'s turn ---")

        position = player_fn(game_state, current_identity)

        if not game_state.is_valid_move(position):
            print(f"  ERROR: Invalid move {position} from {symbol}. Forfeiting.")
            winner = 1 - current_identity
            return {
                "winner": winner,
                "winner_symbol": "X" if winner == 1 else "O",
                "board": game_state.board.copy(),
                "move_count": move_count,
                "reason": "invalid_move",
            }

        game_state.make_move(position, current_identity)
        move_count += 1

        print(f"{symbol} played {position}")
        print(game_state.display())

    winner = game_state.check_winner()
    if winner is not None:
        symbol = "X" if winner == 1 else "O"
        print(f"\n🎉 {symbol} wins!")
        return {
            "winner": winner,
            "winner_symbol": symbol,
            "board": game_state.board.copy(),
            "move_count": move_count,
            "reason": "win",
        }
    else:
        print("\n🤝 It's a draw!")
        return {
            "winner": None,
            "winner_symbol": None,
            "board": game_state.board.copy(),
            "move_count": move_count,
            "reason": "draw",
        }