from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel, Field

SYSTEM_PROMPT_TEMPLATE = """You are a tic-tac-toe player. You play as {symbol} (represented by {value} on the board).

## Board Representation
The board is a list of 9 integers:
- -1 means the position is empty
- 0 means O occupies that position
- 1 means X occupies that position

## Position Layout
 0 | 1 | 2
-----------
 3 | 4 | 5
-----------
 6 | 7 | 8

## Rules
- You MUST use the make_move tool to place your mark
- You can only place on empty positions (value -1)
- You play as {symbol} (value={value}), your opponent plays as {opponent_symbol} (value={opponent_value})
- Call make_move EXACTLY ONCE per turn

## Output Format
Before calling make_move, you MUST first state your reasoning in 2-3 sentences:
1. Which positions you hold and which your opponent holds
2. Whether you can win this turn or need to block
3. Which position you will play and why

Then call make_move once. Do not say anything else after the tool call."""


class MakeMoveInput(BaseModel):
    position: int = Field(
        description="A number from 0 to 8 representing the board position (0-8 grid: 0|1|2/3|4|5/6|7|8)"
    )


@tool(args_schema=MakeMoveInput)
def make_move(position: int) -> str:
    """Place your mark on the tic-tac-toe board."""
    return ""


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
        Tuple of (llm_with_tools, system_prompt). Invoke llm_with_tools with
        system_prompt + user message, then extract the move from
        response.tool_calls[0]["args"]["position"].
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
    llm_with_tools = llm.bind_tools([make_move])

    return llm_with_tools, system_prompt