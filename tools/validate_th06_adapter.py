from __future__ import annotations

import io
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rl.th06_adapter import (  # noqa: E402
    Th06BulletSnapshot,
    Th06EnemySnapshot,
    Th06LaserSnapshot,
    Th06ObservationAdapter,
    Th06PlayerSnapshot,
    Th06ProcessBackend,
    Th06RLEnv,
    Th06Snapshot,
    TH06_RL_PROTOCOL_MAGIC,
    TH06_RL_PROTOCOL_VERSION,
    TH06_RL_REQUEST,
    TH06_RL_RESPONSE,
    TH06_RL_ABI_VERSION,
    TH06_RL_MAGIC,
    _CSnapshot,
    th06_snapshot_byte_size,
)
from rl.reward import local_pccm_cost, wall_proximity  # noqa: E402


class SampleBackend:
    # Return deterministic snapshots without starting the original game.
    def __init__(self) -> None:
        self.snapshot = make_sample_snapshot()
        self.closed = False

    # Reset the sample timeline.
    def reset(self, stage: int, difficulty: int, seed: int) -> Th06Snapshot:
        del seed
        self.snapshot = replace(make_sample_snapshot(), stage=stage, difficulty=difficulty)
        return self.snapshot

    # Advance one sample frame and preserve movement diagnostics.
    def step(self, action: int, focus: bool, shoot: bool) -> Th06Snapshot:
        del focus, shoot
        requested_dx = 1.0 if action in (4, 6, 8) else 0.0
        player = replace(
            self.snapshot.player,
            x=self.snapshot.player.x + requested_dx,
            requested_dx=requested_dx,
            actual_dx=requested_dx,
        )
        self.snapshot = replace(self.snapshot, game_frame=self.snapshot.game_frame + 1, player=player)
        return self.snapshot

    # Mark the sample backend as closed.
    def close(self) -> None:
        self.closed = True


# Build a snapshot that covers bullets, lasers, and enemy contact hitboxes.
def make_sample_snapshot() -> Th06Snapshot:
    return Th06Snapshot(
        game_frame=120,
        stage=1,
        difficulty=1,
        player=Th06PlayerSnapshot(
            x=192.0,
            y=380.0,
            hitbox_width=1.25,
            hitbox_height=1.25,
            state=0,
            lives=3,
            deaths=0,
        ),
        bullets=(
            Th06BulletSnapshot(
                x=192.0,
                y=300.0,
                vx_per_frame=0.0,
                vy_per_frame=2.0,
                hitbox_width=8.0,
                hitbox_height=8.0,
                speed_per_frame=2.0,
                angle=np.pi / 2.0,
                state=1,
                flags=0,
            ),
        ),
        lasers=(
            Th06LaserSnapshot(
                x=80.0,
                y=200.0,
                angle=0.0,
                start_offset=0.0,
                end_offset=120.0,
                width=12.0,
                speed_per_frame=0.0,
                timer_frames=30.0,
                state=1,
                flags=0,
                hitbox_active=True,
            ),
        ),
        enemies=(
            Th06EnemySnapshot(
                x=250.0,
                y=100.0,
                vx_per_frame=1.0,
                vy_per_frame=0.0,
                hitbox_width=24.0,
                hitbox_height=24.0,
                life=100,
                max_life=100,
                is_boss=False,
                contact_active=True,
            ),
        ),
    )


# Build one valid empty C snapshot for transport validation.
def make_sample_snapshot_bytes() -> bytes:
    snapshot = _CSnapshot()
    snapshot.magic = TH06_RL_MAGIC
    snapshot.abi_version = TH06_RL_ABI_VERSION
    snapshot.stage = 1
    snapshot.difficulty = 1
    snapshot.supervisor_state = 2
    snapshot.player.x = 192.0
    snapshot.player.y = 380.0
    snapshot.player.hitbox_width = 1.25
    snapshot.player.hitbox_height = 1.25
    return bytes(snapshot)


# Validate the fixed native process protocol layout.
def validate_process_protocol() -> None:
    if TH06_RL_REQUEST.size != 36 or TH06_RL_RESPONSE.size != 16:
        raise AssertionError("The Python process protocol layout does not match the C++ server.")
    request = TH06_RL_REQUEST.pack(TH06_RL_PROTOCOL_MAGIC, TH06_RL_PROTOCOL_VERSION, 1, 1, 2, 3, 0, 0.0, 0.0)
    magic, version, operation, stage, difficulty, seed, _, _, _ = TH06_RL_REQUEST.unpack(request)
    if (magic, version, operation, stage, difficulty, seed) != (TH06_RL_PROTOCOL_MAGIC, 1, 1, 1, 2, 3):
        raise AssertionError("The native process request was not encoded correctly.")
    snapshot = make_sample_snapshot_bytes()
    if len(snapshot) != th06_snapshot_byte_size():
        raise AssertionError("The sample native snapshot has the wrong size.")
    response = TH06_RL_RESPONSE.pack(
        TH06_RL_PROTOCOL_MAGIC,
        TH06_RL_PROTOCOL_VERSION,
        0,
        len(snapshot),
    )

    class FakeProcess:
        # Provide in-memory pipes with the same interface as Popen.
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(response + snapshot)
            self.returncode = None

        # Report a running child process during the request.
        def poll(self) -> None:
            return None

    backend = object.__new__(Th06ProcessBackend)
    backend.process = FakeProcess()
    parsed = backend._request(1, 1, 1, 0)
    if parsed is None or parsed.stage != 1:
        raise AssertionError("The native process response was not parsed correctly.")


# Validate the environment contract used by the CNN PPO trainer.
def validate_environment() -> None:
    backend = SampleBackend()
    env = Th06RLEnv(backend=backend, max_steps=2, frame_stack=2, frame_stack_interval=1)
    observation = env.reset(seed=7)
    if observation["red_pccm"].shape != (64, 64):
        raise AssertionError("The TH06 environment returned an invalid initial observation.")
    if len(env.get_map_history()) != 2:
        raise AssertionError("The TH06 environment did not initialize frame stacking.")
    _, reward, done, info = env.step(4)
    if not np.isfinite(reward) or done or info["frame_steps"] != 1:
        raise AssertionError("The TH06 environment returned an invalid first step.")
    _, _, done, info = env.step(0)
    if not done or info["frame_steps"] != 2:
        raise AssertionError("The TH06 environment did not honor max_steps.")
    env.close()
    if not backend.closed:
        raise AssertionError("The TH06 environment did not close its backend.")


# Validate masks and margins at the native movement boundaries.
def validate_playable_bounds() -> None:
    adapter = Th06ObservationAdapter()
    sample = replace(make_sample_snapshot(), bullets=(), lasers=(), enemies=())
    left_observation = adapter.build(
        replace(sample, player=replace(sample.player, x=8.0, y=224.0))
    )
    right_observation = adapter.build(
        replace(sample, player=replace(sample.player, x=376.0, y=224.0))
    )
    top_observation = adapter.build(
        replace(sample, player=replace(sample.player, x=192.0, y=16.0))
    )
    bottom_observation = adapter.build(
        replace(sample, player=replace(sample.player, x=192.0, y=432.0))
    )
    center_observation = adapter.build(
        replace(sample, player=replace(sample.player, x=192.0, y=224.0))
    )
    if not np.isclose(left_observation["player_features"][4], 0.0):
        raise AssertionError("The left movement boundary was not encoded as zero margin.")
    if not np.isclose(right_observation["player_features"][5], 0.0):
        raise AssertionError("The right movement boundary was not encoded as zero margin.")
    if not np.isclose(top_observation["player_features"][6], 0.0):
        raise AssertionError("The top movement boundary was not encoded as zero margin.")
    if not np.isclose(bottom_observation["player_features"][7], 0.0):
        raise AssertionError("The bottom movement boundary was not encoded as zero margin.")
    if np.all(left_observation["red_valid"] > 0.0):
        raise AssertionError("The red playable mask did not mark the left unreachable area.")
    if np.all(right_observation["red_valid"] > 0.0):
        raise AssertionError("The red playable mask did not mark the right unreachable area.")
    boundary_observations = (
        left_observation,
        right_observation,
        top_observation,
        bottom_observation,
    )
    if any(wall_proximity(observation) < 0.99 for observation in boundary_observations):
        raise AssertionError("Wall proximity did not use the native movement boundaries.")
    if local_pccm_cost(left_observation) <= local_pccm_cost(center_observation):
        raise AssertionError("PCCM did not place the left wall at the native movement boundary.")


# Validate shapes and finite values expected by the CNN path.
def main() -> None:
    adapter = Th06ObservationAdapter()
    observation = adapter.build(make_sample_snapshot())
    expected_shapes = {
        "blue_density": (8, 8),
        "blue_pccm": (8, 8),
        "blue_valid": (8, 8),
        "yellow_density": (16, 16),
        "yellow_pccm": (16, 16),
        "yellow_valid": (16, 16),
        "red_occupancy": (64, 64),
        "red_pccm": (64, 64),
        "red_valid": (64, 64),
        "player_features": (8,),
    }
    for key, shape in expected_shapes.items():
        value = np.asarray(observation[key])
        if value.shape != shape:
            raise AssertionError(f"{key} has shape {value.shape}, expected {shape}.")
        if not np.all(np.isfinite(value)):
            raise AssertionError(f"{key} contains a non-finite value.")
    if float(np.max(observation["red_pccm"])) <= 0.0:
        raise AssertionError("The sample hazards did not produce red PCCM cost.")
    validate_playable_bounds()
    validate_environment()
    validate_process_protocol()
    print(f"TH06 snapshot ABI bytes: {th06_snapshot_byte_size()}")
    print("TH06 observation adapter validation passed.")


if __name__ == "__main__":
    main()
