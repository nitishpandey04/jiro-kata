from __future__ import annotations

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

SYSTEM_PROMPT_TEMPLATE = """You are a tic-tac-toe player. You play as {symbol} (value={value}). Your opponent plays as {opponent_symbol} (value={opponent_value}).

## Board
A list of 9 integers. Position layout:

 0 | 1 | 2
-----------
 3 | 4 | 5
-----------
 6 | 7 | 8

Values: -1 = empty, 0 = O, 1 = X

## Winning Lines (check these every turn)
Row 0: positions 0, 1, 2
Row 1: positions 3, 4, 5
Row 2: positions 6, 7, 8
Col 0: positions 0, 3, 6
Col 1: positions 1, 4, 7
Col 2: positions 2, 5, 8
Diag:   positions 0, 4, 8
Anti:   positions 2, 4, 6

## Rules
You can only play on empty positions (value -1).

## How to Analyze
Before choosing your move, systematically check EVERY winning line:
1. Do you have 2 marks in any line? If yes and the 3rd position is empty → WIN, play there immediately.
2. Does opponent have 2 marks in any line? If yes and the 3rd position is empty → BLOCK, play there.
3. Otherwise, pick the best available position (center > corners > sides).

Your reasoning must explicitly state which lines you checked and what you found."""


class MoveResponse(BaseModel):
    reasoning: str = Field(
        description="Systematic analysis of the board: list which lines you checked, any winning or blocking opportunities found, and why you chose this position"
    )
    position: int = Field(
        description="The board position (0-8) where you want to place your mark"
    )


def create_jiro_kata_agent(
    identity: int,
    model: str,
) -> tuple:
    """
    Create a Jiro Kata playing agent. The game is tic-tac-toe, but the user-facing name is Jiro Kata.

    Args:
        identity: 1 for X, 0 for O
        model: Model string, e.g. "openai:gpt-4.1-mini"

    Returns:
        Tuple of (structured_model, system_prompt). The model has with_structured_output(MoveResponse)
        applied. Call structured_model.invoke() with system_prompt + user message, then read
        response.reasoning and response.position.
    """
    if identity not in (0, 1):
        raise ValueError(f"Identity must be 0 (O) or 1 (X), got {identity}")

    symbol = "X" if identity == 1 else "O"
    opponent_symbol = "O" if identity == 1 else "X"
    value = identity
    opponent_value = 1 - identity

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        symbol=symbol,
        value=value,
        opponent_symbol=opponent_symbol,
        opponent_value=opponent_value,
    )

    provider, model_name = model.split(":", 1) if ":" in model else ("openai", model)
    llm = init_chat_model(model_name, model_provider=provider)
    structured_model = llm.with_structured_output(MoveResponse)

    return structured_model, system_prompt