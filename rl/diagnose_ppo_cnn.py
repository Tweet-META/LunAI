from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rl.cnn_observation_utils import cnn_observation, cnn_observation_shapes
from rl.ppo_cnn_agent import CNNPPOAgent, load_cnn_ppo_config
from rl.reward import wall_proximity
from rl.touhou_rl_env import TouhouRLEnv
from rl.train_ppo_cnn import validate_checkpoint_shapes


NO_BULLET_POSITIONS = {
    "center": (0.50, 0.50),
    "top": (0.50, 0.08),
    "bottom": (0.50, 0.92),
    "left": (0.08, 0.50),
    "right": (0.92, 0.50),
    "bottom_left": (0.08, 0.92),
    "bottom_right": (0.92, 0.92),
}
DOWN_ACTIONS = (2, 7, 8)


# Calculate entropy from one categorical action distribution.
def action_entropy(probabilities: np.ndarray) -> float:
    safe_probabilities = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    return float(-np.sum(safe_probabilities * np.log(safe_probabilities)))


# Move the player to one normalized playfield position.
def set_player_position(env: TouhouRLEnv, x: float, y: float) -> dict[str, np.ndarray]:
    game_x, game_y, game_width, game_height = env.GAME_ZONE
    world_x = game_x + float(np.clip(x, 0.0, 1.0)) * game_width
    world_y = game_y + float(np.clip(y, 0.0, 1.0)) * game_height
    env.scene.player.position = env.Vector2(world_x, world_y)
    env.scene.player.collider.position = env.scene.player.position
    observation = env.get_observation()
    env.last_observation = observation
    env._reset_map_history(observation)
    return observation


# Create an environment with settings compatible with the saved model.
def create_environment(
    args: argparse.Namespace,
    level_file: str,
    render: bool = False,
) -> TouhouRLEnv:
    return TouhouRLEnv(
        render_mode="human" if render else None,
        max_steps=args.max_steps,
        action_repeat=1,
        level_file=level_file,
        level_spawn_time_jitter=args.level_spawn_time_jitter,
        random_player_start=False,
        frame_stack=args.frame_stack,
        frame_stack_interval=args.frame_stack_interval,
        pccm_prediction_frames=args.pccm_prediction_frames,
        pccm_halo_width=args.pccm_halo_width,
        pccm_wall_margin=args.pccm_wall_margin,
        pccm_upper_field_threshold=args.pccm_upper_field_threshold,
        pccm_upper_field_cost=args.pccm_upper_field_cost,
    )


# Save dictionaries as one CSV file with stable columns.
def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Measure policy probabilities on an empty field at fixed positions.
def diagnose_empty_field(
    args: argparse.Namespace,
    agent: CNNPPOAgent,
    env: TouhouRLEnv,
    output_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    env.reset(seed=args.seed)
    for name, (x, y) in NO_BULLET_POSITIONS.items():
        observation = set_player_position(env, x, y)
        state = cnn_observation(observation, env.get_map_history())
        probabilities = agent.action_probs(state)
        order = np.argsort(probabilities)[::-1]
        top_three = ", ".join(
            f"{env.ACTIONS[int(index)]}:{probabilities[int(index)]:.4f}"
            for index in order[:3]
        )
        row: dict[str, object] = {
            "position": name,
            "x": x,
            "y": y,
            "top_3": top_three,
            "max_probability": float(probabilities[order[0]]),
            "entropy": action_entropy(probabilities),
            "down_probability": float(np.sum(probabilities[list(DOWN_ACTIONS)])),
        }
        for action, action_name in env.ACTIONS.items():
            row[f"prob_{action_name}"] = float(probabilities[action])
        rows.append(row)
        print(
            f"empty position={name:>12} top3=[{top_three}] "
            f"max={row['max_probability']:.4f} entropy={row['entropy']:.4f} "
            f"down_share={row['down_probability']:.4f}"
        )

    write_csv(output_dir / "empty_field_action_probs.csv", rows)
    return rows


# Convert normalized positions into one count heatmap.
def position_heatmap(positions: list[tuple[float, float]], bins: int = 24) -> np.ndarray:
    heatmap = np.zeros((bins, bins), dtype=np.int32)
    for x, y in positions:
        column = min(bins - 1, max(0, int(x * bins)))
        row = min(bins - 1, max(0, int(y * bins)))
        heatmap[row, column] += 1
    return heatmap


# Save a simple blue-to-red player position heatmap.
def save_heatmap(path: Path, heatmap: np.ndarray) -> None:
    scaled = np.log1p(heatmap.astype(np.float32))
    maximum = float(np.max(scaled))
    if maximum > 0.0:
        scaled /= maximum
    red = np.asarray(255.0 * scaled, dtype=np.uint8)
    green = np.asarray(80.0 + 120.0 * scaled, dtype=np.uint8)
    blue = np.asarray(220.0 - 190.0 * scaled, dtype=np.uint8)
    rgb = np.stack((red, green, blue), axis=-1)
    image = Image.fromarray(rgb, mode="RGB").resize((600, 600), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


# Run fixed-level episodes and record movement bias metrics.
def diagnose_episodes(
    args: argparse.Namespace,
    agent: CNNPPOAgent,
    output_dir: Path,
) -> dict[str, object]:
    env = create_environment(args, args.level_file, render=args.render)
    action_counts = np.zeros(len(env.ACTIONS), dtype=np.int64)
    positions: list[tuple[float, float]] = []
    trajectory_rows: list[dict[str, object]] = []
    wall_steps = 0
    total_steps = 0
    collisions = 0
    episode_frames: list[int] = []

    try:
        for episode in range(1, args.episodes + 1):
            observation = env.reset(seed=args.seed + episode)
            state = cnn_observation(observation, env.get_map_history())
            done = False
            while not done:
                if args.stochastic:
                    action, _, _ = agent.select_action(state)
                else:
                    action = agent.select_greedy_action(state)
                observation, reward, done, info = env.step(action)
                state = cnn_observation(observation, env.get_map_history())
                player_x = float(observation["player_features"][0])
                player_y = float(observation["player_features"][1])
                proximity = wall_proximity(observation)
                action_counts[action] += 1
                positions.append((player_x, player_y))
                total_steps += 1
                wall_steps += int(proximity > 0.0)
                trajectory_rows.append(
                    {
                        "episode": episode,
                        "decision_step": info["decision_steps"],
                        "frame_step": info["frame_steps"],
                        "player_x": player_x,
                        "player_y": player_y,
                        "action": action,
                        "action_name": env.ACTIONS[action],
                        "reward": reward,
                        "wall_proximity": proximity,
                        "collided": int(info["collided"]),
                    }
                )
            collisions += int(info["collided"])
            episode_frames.append(int(info["frame_steps"]))
    finally:
        env.close()

    write_csv(output_dir / "trajectory.csv", trajectory_rows)
    save_heatmap(output_dir / "player_position_heatmap.png", position_heatmap(positions))
    down_count = int(np.sum(action_counts[list(DOWN_ACTIONS)]))
    summary: dict[str, object] = {
        "episodes": args.episodes,
        "total_decision_steps": total_steps,
        "mean_frame_steps": float(np.mean(episode_frames)) if episode_frames else 0.0,
        "collisions": collisions,
        "wall_time_ratio": wall_steps / max(1, total_steps),
        "down_action_ratio": down_count / max(1, total_steps),
        "action_counts": {env.ACTIONS[index]: int(count) for index, count in enumerate(action_counts)},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# Build the command line parser for CNN policy diagnosis.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="diagnostics/cnn_policy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--frame-stack", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--frame-stack-interval", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--level-file", type=str, default="level_diagnostic_aimed.json")
    parser.add_argument("--level-spawn-time-jitter", type=float, default=0.0)
    parser.add_argument("--pccm-prediction-frames", type=int, default=5)
    parser.add_argument("--pccm-halo-width", type=float, default=32.0)
    parser.add_argument("--pccm-wall-margin", type=float, default=0.12)
    parser.add_argument("--pccm-upper-field-threshold", type=float, default=0.70)
    parser.add_argument("--pccm-upper-field-cost", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser


# Load one model and run all policy diagnostics.
def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_cnn_ppo_config(str(Path(args.model_path)), device=args.device)
    args.pccm_prediction_frames = config.pccm_prediction_frames
    args.pccm_halo_width = config.pccm_halo_width
    args.pccm_wall_margin = config.pccm_wall_margin
    args.pccm_upper_field_threshold = config.pccm_upper_field_threshold
    args.pccm_upper_field_cost = config.pccm_upper_field_cost

    empty_env = create_environment(args, "level_diagnostic_empty.json", render=args.render)
    try:
        observation = empty_env.reset(seed=args.seed)
        shapes = cnn_observation_shapes(observation, empty_env.get_map_history())
        validate_checkpoint_shapes(
            config,
            shapes,
            args.frame_stack,
            args.frame_stack_interval,
            config.pccm_prediction_frames,
            config.pccm_halo_width,
            config.pccm_wall_margin,
            config.pccm_upper_field_threshold,
            config.pccm_upper_field_cost,
        )

        agent = CNNPPOAgent(config)
        agent.load(str(Path(args.model_path)))
        diagnose_empty_field(args, agent, empty_env, output_dir)
        diagnose_episodes(args, agent, output_dir)
    finally:
        empty_env.close()


if __name__ == "__main__":
    main()
