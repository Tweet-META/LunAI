from __future__ import annotations

import multiprocessing as mp
import traceback
from collections.abc import Sequence
from typing import Any

from rl.cnn_observation_utils import CNNObservation, cnn_observation
from rl.touhou_rl_env import TouhouRLEnv


# Run one headless Touhou environment in a separate CPU process.
def touhou_env_worker(connection: Any, env_kwargs: dict[str, Any]) -> None:
    env: TouhouRLEnv | None = None
    try:
        env = TouhouRLEnv(**env_kwargs)
        while True:
            command, payload = connection.recv()
            if command == "reset":
                observation = env.reset(seed=int(payload))
                state = cnn_observation(observation, env.get_map_history())
                connection.send(("ok", state))
            elif command == "step":
                observation, reward, done, info = env.step(int(payload))
                state = cnn_observation(observation, env.get_map_history())
                connection.send(("ok", (state, float(reward), bool(done), info)))
            elif command == "close":
                connection.send(("ok", None))
                break
            else:
                raise ValueError(f"Unknown environment command: {command}")
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class ParallelTouhouEnvs:
    # Create several headless Touhou environments for one shared policy.
    def __init__(self, num_envs: int, env_kwargs: dict[str, Any]):
        if num_envs < 2:
            raise ValueError("ParallelTouhouEnvs needs at least two environments.")

        context = mp.get_context("spawn")
        self.connections: list[Any] = []
        self.processes: list[mp.Process] = []
        for _ in range(num_envs):
            parent_connection, child_connection = context.Pipe()
            process = context.Process(
                target=touhou_env_worker,
                args=(child_connection, env_kwargs),
            )
            process.start()
            child_connection.close()
            self.connections.append(parent_connection)
            self.processes.append(process)

    # Reset every environment with its own random seed.
    def reset(self, seeds: Sequence[int]) -> list[CNNObservation]:
        if len(seeds) != len(self.connections):
            raise ValueError("The seed count must match the environment count.")
        for connection, seed in zip(self.connections, seeds):
            connection.send(("reset", int(seed)))
        return [self._receive(connection) for connection in self.connections]

    # Reset selected environments after their episodes end.
    def reset_indices(self, seed_by_index: dict[int, int]) -> dict[int, CNNObservation]:
        for index, seed in seed_by_index.items():
            self.connections[index].send(("reset", int(seed)))
        return {
            index: self._receive(self.connections[index])
            for index in seed_by_index
        }

    # Advance every environment once with one action per environment.
    def step(
        self,
        actions: Sequence[int],
    ) -> list[tuple[CNNObservation, float, bool, dict[str, Any]]]:
        if len(actions) != len(self.connections):
            raise ValueError("The action count must match the environment count.")
        for connection, action in zip(self.connections, actions):
            connection.send(("step", int(action)))
        return [self._receive(connection) for connection in self.connections]

    # Stop worker processes and release their pygame resources.
    def close(self) -> None:
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass

        for connection in self.connections:
            try:
                self._receive(connection)
            except (BrokenPipeError, EOFError, OSError, RuntimeError):
                pass
            connection.close()

        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

    # Receive one worker response and raise worker errors in the main process.
    def _receive(self, connection: Any) -> Any:
        status, payload = connection.recv()
        if status == "error":
            raise RuntimeError(f"Touhou environment worker failed:\n{payload}")
        return payload
