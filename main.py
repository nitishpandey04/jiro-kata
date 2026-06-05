from __future__ import annotations

import argparse

from dotenv import load_dotenv

from agent import create_jiro_kata_agent
from game import GameState, agent_player_fn, human_player, play_game

load_dotenv()

DEFAULT_MODEL = "openai:gpt-4.1-mini"


def create_agent_player(identity: int, model: str, game_state: GameState):
    llm_with_tools, system_prompt = create_jiro_kata_agent(
        identity=identity,
        model=model,
    )

    def player_fn(gs: GameState, pid: int) -> int:
        return agent_player_fn(llm_with_tools, system_prompt, gs, pid)

    return player_fn


def main():
    parser = argparse.ArgumentParser(description="Jiro Kata: Agent vs Human or Agent vs Agent")
    parser.add_argument(
        "--mode",
        choices=["agent-vs-human", "agent-vs-agent"],
        default="agent-vs-human",
        help="Game mode (default: agent-vs-human)",
    )
    parser.add_argument(
        "--x-model",
        default=DEFAULT_MODEL,
        help=f"Model for player X (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--o-model",
        default=DEFAULT_MODEL,
        help=f"Model for player O (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--human-plays",
        choices=["X", "O"],
        default="O",
        help="Which side the human plays in agent-vs-human mode (default: O)",
    )
    args = parser.parse_args()

    game_state = GameState()

    if args.mode == "agent-vs-human":
        human_identity = 1 if args.human_plays == "X" else 0
        agent_identity = 1 - human_identity
        agent_model = args.x_model if agent_identity == 1 else args.o_model

        agent_fn = create_agent_player(agent_identity, agent_model, game_state)

        if human_identity == 1:
            player_x = human_player
            player_o = agent_fn
            print(f"You are X (go first). Agent is O (model: {agent_model})")
        else:
            player_x = agent_fn
            player_o = human_player
            print(f"Agent is X (goes first, model: {agent_model}). You are O")

        result = play_game(player_x, player_o, game_state)

    elif args.mode == "agent-vs-agent":
        x_fn = create_agent_player(1, args.x_model, game_state)
        o_fn = create_agent_player(0, args.o_model, game_state)

        print(f"Agent X model: {args.x_model}")
        print(f"Agent O model: {args.o_model}")

        result = play_game(x_fn, o_fn, game_state)

    print("\n" + "=" * 40)
    print("GAME OVER")
    print(f"Board: {game_state.board}")
    if result["winner"] is not None:
        print(f"Winner: {result['winner_symbol']}")
    else:
        print("Result: Draw")
    print(f"Total moves: {result['move_count']}")

    return result


if __name__ == "__main__":
    main()