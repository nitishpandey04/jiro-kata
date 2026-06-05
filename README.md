# Jiro Kata

Two AI agents walk into a tic-tac-toe board. Both leave disappointed.

## What is this?

Jiro Kata is tic-tac-toe, but played by AI agents. Set up two LLMs against each other, or play against one yourself. Watch them reason, strategize, and occasionally blunder — just like humans do.

## Why "Jiro Kata"?

Because tic-tac-toe deserved a fancier name.

## How it works

- The board is a list of 9 integers: `-1` (empty), `0` (O), `1` (X)
- Each agent uses structured output (`MoveResponse`) with `reasoning` + `position` fields
- Agents systematically check all 8 winning lines before choosing a move
- Game state is shared — both players update the same board

## Setup

```bash
# Requires uv (https://docs.astral.sh/uv/)
uv sync
```

Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

## Play

**Agent vs Agent:**
```bash
uv run python main.py --mode agent-vs-agent
```

**Agent vs You:**
```bash
uv run python main.py --mode agent-vs-human
```

**Asymmetric models (where the fun begins):**
```bash
uv run python main.py --mode agent-vs-agent --x-model openai:gpt-4.1-nano --o-model openai:gpt-4.1-mini
```

Two gpt-4.1-mini agents will draw every time. It's a solved game. But throw a weaker model in there and things get interesting.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `agent-vs-human` | `agent-vs-agent` or `agent-vs-human` |
| `--x-model` | `openai:gpt-4.1-mini` | Model for player X |
| `--o-model` | `openai:gpt-4.1-mini` | Model for player O |
| `--human-plays` | `O` | Which side you play in human mode (`X` or `O`) |

## Architecture

```
agent.py    - LLM agent creation (structured output, system prompt)
game.py     - GameState, display, game loop, human input
main.py     - CLI entry point
```

## Screenshot

```
--- X's turn ---
Checked all winning lines for a winning move: I have positions 4, 8 —
anti-diagonal (2, 4, 6) has X at 4, no immediate win. Row 1 (3, 4, 5)
has X at 4, no immediate win. Must block: opponent has positions 0, 3
in column 0, position 6 is empty — blocking at position 6.
─────────────────
X played 6
```

## License

MIT