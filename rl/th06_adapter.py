from __future__ import annotations

import ctypes
import struct
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import numpy as np

from observation_builder import BulletState, ObservationBuilder, ObservationConfig, PlayerState
from rl.cnn_observation_utils import cnn_map_keys
from rl.reward import blocked_movement_ratio, compute_frame_reward, local_pccm_cost, wall_proximity


TH06_RL_MAGIC = 0x4C523654
TH06_RL_ABI_VERSION = 1
TH06_FPS = 60.0
TH06_PLAYFIELD_WIDTH = 384
TH06_PLAYFIELD_HEIGHT = 448
TH06_PLAYABLE_BOUNDS = (8.0, 16.0, 376.0, 432.0)
TH06_YELLOW_SIZE = (204, 204)
TH06_RED_SIZE = (64, 64)
TH06_DEFAULT_PCCM_HALO_WIDTH = 20.0
TH06_MAX_BULLETS = 640
TH06_MAX_LASERS = 64
TH06_MAX_ENEMIES = 256
TH06_RL_PROTOCOL_MAGIC = 0x50524C36
TH06_RL_PROTOCOL_VERSION = 1
TH06_RL_SERVER_RESET = 1
TH06_RL_SERVER_STEP = 2
TH06_RL_SERVER_SET_PLAYER = 3
TH06_RL_SERVER_SNAPSHOT = 4
TH06_RL_SERVER_CLOSE = 5
TH06_RL_REQUEST = struct.Struct("<IIIiiiiff")
TH06_RL_RESPONSE = struct.Struct("<IIiI")


@dataclass(frozen=True)
class Th06PlayerSnapshot:
    x: float
    y: float
    hitbox_width: float
    hitbox_height: float
    state: int
    lives: int
    deaths: int
    requested_dx: float = 0.0
    requested_dy: float = 0.0
    actual_dx: float = 0.0
    actual_dy: float = 0.0


@dataclass(frozen=True)
class Th06BulletSnapshot:
    x: float
    y: float
    vx_per_frame: float
    vy_per_frame: float
    hitbox_width: float
    hitbox_height: float
    speed_per_frame: float
    angle: float
    state: int
    flags: int


@dataclass(frozen=True)
class Th06LaserSnapshot:
    x: float
    y: float
    angle: float
    start_offset: float
    end_offset: float
    width: float
    speed_per_frame: float
    timer_frames: float
    state: int
    flags: int
    hitbox_active: bool


@dataclass(frozen=True)
class Th06EnemySnapshot:
    x: float
    y: float
    vx_per_frame: float
    vy_per_frame: float
    hitbox_width: float
    hitbox_height: float
    life: int
    max_life: int
    is_boss: bool
    contact_active: bool


@dataclass(frozen=True)
class Th06Snapshot:
    game_frame: int
    stage: int
    difficulty: int
    player: Th06PlayerSnapshot
    supervisor_state: int = 2
    game_completed: bool = False
    bullets: tuple[Th06BulletSnapshot, ...] = ()
    lasers: tuple[Th06LaserSnapshot, ...] = ()
    enemies: tuple[Th06EnemySnapshot, ...] = ()


class Th06Backend(Protocol):
    # Reset one original TH06 stage and return its first state.
    def reset(self, stage: int, difficulty: int, seed: int) -> Th06Snapshot: ...

    # Advance exactly one original TH06 game frame.
    def step(self, action: int, focus: bool, shoot: bool) -> Th06Snapshot: ...

    # Release resources held by the backend.
    def close(self) -> None: ...


class Th06ProcessBackend:
    # Start the standalone TH06 logic server without creating a window.
    def __init__(self, executable: str | Path, assets_dir: str | Path | None = None):
        self.executable = Path(executable).expanduser().resolve()
        if not self.executable.is_file():
            raise FileNotFoundError(f"TH06 RL server was not found: {self.executable}")
        self.working_directory = (
            Path(assets_dir).expanduser().resolve()
            if assets_dir is not None and str(assets_dir)
            else self.executable.parent
        )
        if not self.working_directory.is_dir():
            raise FileNotFoundError(f"TH06 assets directory was not found: {self.working_directory}")
        self.creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self.has_reset = False
        self.process = self._start_process()

    # Start one clean native game process.
    def _start_process(self) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [str(self.executable)],
            cwd=str(self.working_directory),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
            creationflags=self.creation_flags,
        )

    # Replace the native process before a new episode.
    def _restart_process(self) -> None:
        self.close()
        self.process = self._start_process()

    # Read one exact protocol record or report an early server exit.
    def _read_exact(self, stream: BinaryIO, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = stream.read(remaining)
            if not chunk:
                code = self.process.poll()
                if code is None:
                    try:
                        code = self.process.wait(timeout=0.25)
                    except subprocess.TimeoutExpired:
                        pass
                raise RuntimeError(f"TH06 RL server closed its output early (exit code {code}).")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    # Exchange one binary command with the standalone server.
    def _request(
        self,
        operation: int,
        arg0: int = 0,
        arg1: int = 0,
        arg2: int = 0,
        arg3: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        expect_snapshot: bool = True,
    ) -> Th06Snapshot | None:
        if self.process.poll() is not None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"TH06 RL server is not running (exit code {self.process.returncode}).")
        request = TH06_RL_REQUEST.pack(
            TH06_RL_PROTOCOL_MAGIC,
            TH06_RL_PROTOCOL_VERSION,
            int(operation),
            int(arg0),
            int(arg1),
            int(arg2),
            int(arg3),
            float(x),
            float(y),
        )
        self.process.stdin.write(request)
        self.process.stdin.flush()
        header = self._read_exact(self.process.stdout, TH06_RL_RESPONSE.size)
        magic, version, status, snapshot_size = TH06_RL_RESPONSE.unpack(header)
        if magic != TH06_RL_PROTOCOL_MAGIC or version != TH06_RL_PROTOCOL_VERSION:
            raise RuntimeError("TH06 RL server returned an incompatible protocol header.")
        if status != 0:
            raise RuntimeError(f"TH06 RL server command {operation} failed with status {status}.")
        if not expect_snapshot:
            if snapshot_size != 0:
                self._read_exact(self.process.stdout, snapshot_size)
            return None
        expected_size = th06_snapshot_byte_size()
        if snapshot_size != expected_size:
            raise RuntimeError(f"TH06 RL server returned {snapshot_size} snapshot bytes; expected {expected_size}.")
        return snapshot_from_bytes(self._read_exact(self.process.stdout, snapshot_size))

    # Reset one original TH06 stage in the native process.
    def reset(self, stage: int, difficulty: int, seed: int) -> Th06Snapshot:
        if self.has_reset:
            self._restart_process()
        signed_seed = ctypes.c_int32(int(seed)).value
        snapshot = self._request(TH06_RL_SERVER_RESET, stage, difficulty, signed_seed)
        if snapshot is None:
            raise RuntimeError("TH06 RL reset returned no snapshot.")
        self.has_reset = True
        return snapshot

    # Advance exactly one original TH06 game frame.
    def step(self, action: int, focus: bool, shoot: bool) -> Th06Snapshot:
        snapshot = self._request(TH06_RL_SERVER_STEP, action, int(focus), int(shoot))
        if snapshot is None:
            raise RuntimeError("TH06 RL step returned no snapshot.")
        return snapshot

    # Move the player for deterministic diagnostics without advancing a frame.
    def set_player_position(self, x: float, y: float) -> Th06Snapshot:
        snapshot = self._request(TH06_RL_SERVER_SET_PLAYER, x=x, y=y)
        if snapshot is None:
            raise RuntimeError("TH06 RL position command returned no snapshot.")
        return snapshot

    # Stop the child process and release its pipes.
    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._request(TH06_RL_SERVER_CLOSE, expect_snapshot=False)
            except (BrokenPipeError, RuntimeError):
                pass
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=2.0)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()


class _CPlayerSnapshot(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("hitbox_width", ctypes.c_float),
        ("hitbox_height", ctypes.c_float),
        ("state", ctypes.c_int32),
        ("lives", ctypes.c_int32),
        ("deaths", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("requested_dx", ctypes.c_float),
        ("requested_dy", ctypes.c_float),
        ("actual_dx", ctypes.c_float),
        ("actual_dy", ctypes.c_float),
    ]


class _CBulletSnapshot(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("vx_per_frame", ctypes.c_float),
        ("vy_per_frame", ctypes.c_float),
        ("hitbox_width", ctypes.c_float),
        ("hitbox_height", ctypes.c_float),
        ("speed_per_frame", ctypes.c_float),
        ("angle", ctypes.c_float),
        ("state", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class _CLaserSnapshot(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("angle", ctypes.c_float),
        ("start_offset", ctypes.c_float),
        ("end_offset", ctypes.c_float),
        ("width", ctypes.c_float),
        ("speed_per_frame", ctypes.c_float),
        ("timer_frames", ctypes.c_float),
        ("state", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("hitbox_active", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _CEnemySnapshot(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("vx_per_frame", ctypes.c_float),
        ("vy_per_frame", ctypes.c_float),
        ("hitbox_width", ctypes.c_float),
        ("hitbox_height", ctypes.c_float),
        ("life", ctypes.c_int32),
        ("max_life", ctypes.c_int32),
        ("is_boss", ctypes.c_uint32),
        ("contact_active", ctypes.c_uint32),
    ]


class _CSnapshot(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("game_frame", ctypes.c_uint32),
        ("stage", ctypes.c_uint32),
        ("difficulty", ctypes.c_uint32),
        ("bullet_count", ctypes.c_uint32),
        ("laser_count", ctypes.c_uint32),
        ("enemy_count", ctypes.c_uint32),
        ("supervisor_state", ctypes.c_uint32),
        ("game_completed", ctypes.c_uint32),
        ("player", _CPlayerSnapshot),
        ("bullets", _CBulletSnapshot * TH06_MAX_BULLETS),
        ("lasers", _CLaserSnapshot * TH06_MAX_LASERS),
        ("enemies", _CEnemySnapshot * TH06_MAX_ENEMIES),
    ]


# Return the fixed byte size exported by the TH06 bridge.
def th06_snapshot_byte_size() -> int:
    return ctypes.sizeof(_CSnapshot)


# Parse one fixed-layout C ABI snapshot.
def snapshot_from_bytes(data: bytes | bytearray | memoryview) -> Th06Snapshot:
    raw = bytes(data)
    expected_size = th06_snapshot_byte_size()
    if len(raw) != expected_size:
        raise ValueError(f"TH06 snapshot must contain {expected_size} bytes, got {len(raw)}.")
    snapshot = _CSnapshot.from_buffer_copy(raw)
    if snapshot.magic != TH06_RL_MAGIC:
        raise ValueError(f"Invalid TH06 snapshot magic 0x{snapshot.magic:08x}.")
    if snapshot.abi_version != TH06_RL_ABI_VERSION:
        raise ValueError(
            f"Unsupported TH06 RL ABI {snapshot.abi_version}; expected {TH06_RL_ABI_VERSION}."
        )
    if snapshot.bullet_count > TH06_MAX_BULLETS:
        raise ValueError("TH06 snapshot bullet count exceeds the ABI capacity.")
    if snapshot.laser_count > TH06_MAX_LASERS:
        raise ValueError("TH06 snapshot laser count exceeds the ABI capacity.")
    if snapshot.enemy_count > TH06_MAX_ENEMIES:
        raise ValueError("TH06 snapshot enemy count exceeds the ABI capacity.")
    return _snapshot_from_c(snapshot)


# Convert a ctypes snapshot into immutable Python values.
def _snapshot_from_c(snapshot: _CSnapshot) -> Th06Snapshot:
    player = snapshot.player
    bullets = tuple(
        Th06BulletSnapshot(
            x=float(item.x),
            y=float(item.y),
            vx_per_frame=float(item.vx_per_frame),
            vy_per_frame=float(item.vy_per_frame),
            hitbox_width=float(item.hitbox_width),
            hitbox_height=float(item.hitbox_height),
            speed_per_frame=float(item.speed_per_frame),
            angle=float(item.angle),
            state=int(item.state),
            flags=int(item.flags),
        )
        for item in snapshot.bullets[: snapshot.bullet_count]
    )
    lasers = tuple(
        Th06LaserSnapshot(
            x=float(item.x),
            y=float(item.y),
            angle=float(item.angle),
            start_offset=float(item.start_offset),
            end_offset=float(item.end_offset),
            width=float(item.width),
            speed_per_frame=float(item.speed_per_frame),
            timer_frames=float(item.timer_frames),
            state=int(item.state),
            flags=int(item.flags),
            hitbox_active=bool(item.hitbox_active),
        )
        for item in snapshot.lasers[: snapshot.laser_count]
    )
    enemies = tuple(
        Th06EnemySnapshot(
            x=float(item.x),
            y=float(item.y),
            vx_per_frame=float(item.vx_per_frame),
            vy_per_frame=float(item.vy_per_frame),
            hitbox_width=float(item.hitbox_width),
            hitbox_height=float(item.hitbox_height),
            life=int(item.life),
            max_life=int(item.max_life),
            is_boss=bool(item.is_boss),
            contact_active=bool(item.contact_active),
        )
        for item in snapshot.enemies[: snapshot.enemy_count]
    )
    return Th06Snapshot(
        game_frame=int(snapshot.game_frame),
        stage=int(snapshot.stage),
        difficulty=int(snapshot.difficulty),
        player=Th06PlayerSnapshot(
            x=float(player.x),
            y=float(player.y),
            hitbox_width=float(player.hitbox_width),
            hitbox_height=float(player.hitbox_height),
            state=int(player.state),
            lives=int(player.lives),
            deaths=int(player.deaths),
            requested_dx=float(player.requested_dx),
            requested_dy=float(player.requested_dy),
            actual_dx=float(player.actual_dx),
            actual_dy=float(player.actual_dy),
        ),
        bullets=bullets,
        lasers=lasers,
        enemies=enemies,
        supervisor_state=int(snapshot.supervisor_state),
        game_completed=bool(snapshot.game_completed),
    )


class Th06ObservationAdapter:
    # Build the existing LunAI maps from original TH06 world state.
    def __init__(
        self,
        pccm_prediction_frames: int = 5,
        pccm_halo_width: float = TH06_DEFAULT_PCCM_HALO_WIDTH,
        pccm_wall_margin: float = 0.12,
        pccm_upper_field_threshold: float = 0.70,
        pccm_upper_field_cost: float = 0.30,
        laser_sample_spacing: float = 4.0,
    ):
        if laser_sample_spacing <= 0.0:
            raise ValueError("Laser sample spacing must be positive.")
        self.laser_sample_spacing = float(laser_sample_spacing)
        self.builder = ObservationBuilder(
            ObservationConfig(
                playfield_width=TH06_PLAYFIELD_WIDTH,
                playfield_height=TH06_PLAYFIELD_HEIGHT,
                playable_bounds=TH06_PLAYABLE_BOUNDS,
                blue_grid=(8, 8),
                yellow_size=TH06_YELLOW_SIZE,
                yellow_grid=(16, 16),
                red_size=TH06_RED_SIZE,
                red_map=(64, 64),
                pccm_prediction_frames=int(pccm_prediction_frames),
                pccm_halo_width=float(pccm_halo_width),
                pccm_wall_margin=float(pccm_wall_margin),
                pccm_upper_field_threshold=float(pccm_upper_field_threshold),
                pccm_upper_field_cost=float(pccm_upper_field_cost),
            )
        )

    # Convert one engine snapshot into the standard observation dictionary.
    def build(self, snapshot: Th06Snapshot, previous_action: int = 0) -> dict[str, np.ndarray]:
        hazards = self._bullet_hazards(snapshot.bullets)
        hazards.extend(self._laser_hazards(snapshot.lasers))
        hazards.extend(self._enemy_hazards(snapshot.enemies))
        player = snapshot.player
        player_state = PlayerState(
            x=player.x,
            y=player.y,
            radius=max(0.1, max(player.hitbox_width, player.hitbox_height) / 2.0),
            previous_action=int(previous_action),
            half_width=max(0.1, player.hitbox_width / 2.0),
            half_height=max(0.1, player.hitbox_height / 2.0),
        )
        return self.builder.build(hazards, player_state)

    # Convert active bullet hitboxes and per-frame velocity units.
    def _bullet_hazards(self, bullets: tuple[Th06BulletSnapshot, ...]) -> list[BulletState]:
        return [
            BulletState(
                x=bullet.x,
                y=bullet.y,
                radius=max(0.1, max(bullet.hitbox_width, bullet.hitbox_height) / 2.0),
                vx=bullet.vx_per_frame * TH06_FPS,
                vy=bullet.vy_per_frame * TH06_FPS,
                half_width=max(0.1, bullet.hitbox_width / 2.0),
                half_height=max(0.1, bullet.hitbox_height / 2.0),
            )
            for bullet in bullets
            if bullet.state == 1
        ]

    # Approximate each active laser rectangle with overlapping circular samples.
    def _laser_hazards(self, lasers: tuple[Th06LaserSnapshot, ...]) -> list[BulletState]:
        hazards: list[BulletState] = []
        for laser in lasers:
            if not laser.hitbox_active or laser.end_offset <= laser.start_offset:
                continue
            radius = max(0.5, laser.width / 2.0)
            spacing = min(self.laser_sample_spacing, radius)
            length = laser.end_offset - laser.start_offset
            sample_count = max(2, int(np.ceil(length / spacing)) + 1)
            offsets = np.linspace(laser.start_offset, laser.end_offset, sample_count)
            cosine = float(np.cos(laser.angle))
            sine = float(np.sin(laser.angle))
            velocity = laser.speed_per_frame * TH06_FPS
            for offset in offsets:
                hazards.append(
                    BulletState(
                        x=laser.x + cosine * float(offset),
                        y=laser.y + sine * float(offset),
                        radius=radius,
                        vx=cosine * velocity,
                        vy=sine * velocity,
                    )
                )
        return hazards

    # Add only enemies whose original engine contact collision is active.
    def _enemy_hazards(self, enemies: tuple[Th06EnemySnapshot, ...]) -> list[BulletState]:
        return [
            BulletState(
                x=enemy.x,
                y=enemy.y,
                radius=max(0.1, max(enemy.hitbox_width, enemy.hitbox_height) / 3.0),
                vx=enemy.vx_per_frame * TH06_FPS,
                vy=enemy.vy_per_frame * TH06_FPS,
                half_width=max(0.1, enemy.hitbox_width / 3.0),
                half_height=max(0.1, enemy.hitbox_height / 3.0),
            )
            for enemy in enemies
            if enemy.contact_active
        ]


class Th06RLEnv:
    ACTIONS = {
        0: "stay",
        1: "up",
        2: "down",
        3: "left",
        4: "right",
        5: "up_left",
        6: "up_right",
        7: "down_left",
        8: "down_right",
    }

    # Wrap a TH06 backend with the current LunAI environment contract.
    def __init__(
        self,
        backend: Th06Backend,
        stage: int = 1,
        difficulty: int = 1,
        max_steps: int | None = None,
        frame_stack: int = 1,
        frame_stack_interval: int = 1,
        focus: bool = True,
        shoot: bool = True,
        render_mode: str | None = None,
        render_fps: int | None = 60,
        render_debug: bool = False,
        observation_adapter: Th06ObservationAdapter | None = None,
    ):
        if not 1 <= frame_stack <= 5:
            raise ValueError("frame_stack must be in 1..5.")
        if not 1 <= frame_stack_interval <= 5:
            raise ValueError("frame_stack_interval must be in 1..5.")
        self.backend = backend
        self.stage = int(stage)
        self.difficulty = int(difficulty)
        self.max_steps = max_steps
        self.frame_stack = int(frame_stack)
        self.frame_stack_interval = int(frame_stack_interval)
        self.map_history_size = 1 + (self.frame_stack - 1) * self.frame_stack_interval
        self.map_history: deque[dict[str, np.ndarray]] = deque(maxlen=self.map_history_size)
        self.map_keys = cnn_map_keys()
        self.focus = bool(focus)
        self.shoot = bool(shoot)
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_debug = bool(render_debug)
        self.adapter = observation_adapter or Th06ObservationAdapter()
        self.snapshot: Th06Snapshot | None = None
        self.last_observation: dict[str, np.ndarray] | None = None
        self.previous_action = 0
        self.steps = 0
        self.episode_reward = 0.0
        self.episode_pccm_cost_sum = 0.0
        self.episode_blocked_movement_ratio_sum = 0.0
        self.episode_wall_frames = 0
        self.episode_action_counts = np.zeros(len(self.ACTIONS), dtype=np.int64)
        self._render_screen = None
        self._render_clock = None
        self._render_closed = False
        self._projection_overlay = None
        self._projection_scaled_overlay = None
        self._projection_heatmaps: dict[str, Any] = {}
        self._projection_scaled_heatmaps: dict[str, Any] = {}

    # Start a deterministic original-game episode.
    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        actual_seed = 0 if seed is None else int(seed)
        self.snapshot = self.backend.reset(self.stage, self.difficulty, actual_seed)
        self.previous_action = 0
        self.steps = 0
        self.episode_reward = 0.0
        self.episode_pccm_cost_sum = 0.0
        self.episode_blocked_movement_ratio_sum = 0.0
        self.episode_wall_frames = 0
        self.episode_action_counts.fill(0)
        self.last_observation = self.adapter.build(self.snapshot, self.previous_action)
        self.map_history.clear()
        for _ in range(self.map_history_size):
            self._append_map_snapshot(self.last_observation)
        self.render()
        return self.last_observation

    # Apply one action to exactly one original TH06 game frame.
    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if action not in self.ACTIONS:
            raise ValueError(f"Action must be in 0..8, got {action}.")
        if self.snapshot is None or self.last_observation is None:
            self.reset()
        if self.snapshot is None:
            raise RuntimeError("TH06 backend did not return an initial snapshot.")

        previous_snapshot = self.snapshot
        self.snapshot = self.backend.step(int(action), self.focus, self.shoot)
        observation = self.adapter.build(self.snapshot, int(action))
        blocked_ratio = blocked_movement_ratio(
            self.snapshot.player.requested_dx,
            self.snapshot.player.requested_dy,
            self.snapshot.player.actual_dx,
            self.snapshot.player.actual_dy,
        )
        collided = self.snapshot.player.deaths > previous_snapshot.player.deaths
        reward = compute_frame_reward(
            observation,
            int(action),
            self.previous_action,
            collided,
            blocked_ratio,
        )
        self.previous_action = int(action)
        self.steps += 1
        self.episode_reward += reward
        local_cost = local_pccm_cost(observation)
        current_wall_proximity = wall_proximity(observation)
        self.episode_pccm_cost_sum += local_cost
        self.episode_blocked_movement_ratio_sum += blocked_ratio
        self.episode_wall_frames += int(current_wall_proximity > 0.0)
        self.episode_action_counts[int(action)] += 1
        self.last_observation = observation
        self._append_map_snapshot(observation)
        stage_finished = self.snapshot.supervisor_state != 2 or self.snapshot.game_completed
        done = collided or stage_finished or (self.max_steps is not None and self.steps >= self.max_steps)
        info = {
            "stage": self.snapshot.stage,
            "difficulty": self.snapshot.difficulty,
            "game_frame": self.snapshot.game_frame,
            "frame_steps": self.steps,
            "decision_steps": self.steps,
            "collided": collided,
            "stage_finished": stage_finished,
            "hp": self.snapshot.player.lives,
            "bullets": len(self.snapshot.bullets),
            "lasers": len(self.snapshot.lasers),
            "enemies": len(self.snapshot.enemies),
            "local_pccm_cost": local_cost,
            "mean_local_pccm": self.episode_pccm_cost_sum / max(1, self.steps),
            "blocked_movement_ratio": blocked_ratio,
            "mean_blocked_movement_ratio": self.episode_blocked_movement_ratio_sum / max(1, self.steps),
            "wall_proximity": current_wall_proximity,
            "wall_time_ratio": self.episode_wall_frames / max(1, self.steps),
            "action_counts": self.episode_action_counts.tolist(),
        }
        self.render()
        if self._render_closed:
            done = True
            info["window_closed"] = True
        return observation, reward, done, info

    # Return the frames selected by the configured stack interval.
    def get_map_history(self) -> tuple[dict[str, np.ndarray], ...]:
        if len(self.map_history) != self.map_history_size:
            raise RuntimeError("Map history is not initialized. Call reset() first.")
        return tuple(self.map_history)[:: self.frame_stack_interval]

    # Store only map tensors used by the CNN.
    def _append_map_snapshot(self, observation: dict[str, np.ndarray]) -> None:
        self.map_history.append(
            {key: np.asarray(observation[key], dtype=np.float32).copy() for key in self.map_keys}
        )

    # Create the optional state renderer only when it is requested.
    def _ensure_renderer(self) -> None:
        if self.render_mode != "human" or self._render_screen is not None:
            return
        import pygame

        pygame.init()
        playfield_scale = 1.5
        playfield_width = int(TH06_PLAYFIELD_WIDTH * playfield_scale)
        playfield_height = int(TH06_PLAYFIELD_HEIGHT * playfield_scale)
        side_width = 300 if self.render_debug else 0
        self._render_screen = pygame.display.set_mode((playfield_width + side_width, playfield_height))
        pygame.display.set_caption("LunAI - native TH06 state")
        self._render_clock = pygame.time.Clock()

    # Draw one engine snapshot without changing the simulation.
    def render(self) -> None:
        if self.render_mode != "human" or self.snapshot is None:
            return
        self._ensure_renderer()
        if self._render_screen is None:
            return
        import pygame

        scale = 1.5
        screen = self._render_screen
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self._render_closed = True

        screen.fill((14, 17, 24))
        playfield = pygame.Rect(0, 0, int(TH06_PLAYFIELD_WIDTH * scale), int(TH06_PLAYFIELD_HEIGHT * scale))
        pygame.draw.rect(screen, (20, 26, 38), playfield)
        pygame.draw.rect(screen, (110, 124, 150), playfield, 2)

        if self.last_observation is not None:
            self._draw_pccm_projection(screen, scale)

        player = self.snapshot.player
        player_half_width = max(0.1, player.hitbox_width / 2.0)
        player_half_height = max(0.1, player.hitbox_height / 2.0)
        for enemy in self.snapshot.enemies:
            if not enemy.contact_active:
                body = pygame.Rect(
                    round((enemy.x - enemy.hitbox_width / 2.0) * scale),
                    round((enemy.y - enemy.hitbox_height / 2.0) * scale),
                    max(1, round(enemy.hitbox_width * scale)),
                    max(1, round(enemy.hitbox_height * scale)),
                )
                pygame.draw.rect(screen, (132, 72, 82), body, 1)
                continue
            enemy_half_width = max(0.1, enemy.hitbox_width / 3.0)
            enemy_half_height = max(0.1, enemy.hitbox_height / 3.0)
            hitbox = pygame.Rect(
                round((enemy.x - enemy_half_width) * scale),
                round((enemy.y - enemy_half_height) * scale),
                max(1, round(enemy_half_width * 2.0 * scale)),
                max(1, round(enemy_half_height * 2.0 * scale)),
            )
            expanded = pygame.Rect(
                round((enemy.x - enemy_half_width - player_half_width) * scale),
                round((enemy.y - enemy_half_height - player_half_height) * scale),
                max(1, round((enemy_half_width + player_half_width) * 2.0 * scale)),
                max(1, round((enemy_half_height + player_half_height) * 2.0 * scale)),
            )
            pygame.draw.rect(screen, (235, 92, 102), hitbox)
            pygame.draw.rect(screen, (255, 172, 178), expanded, 1)
        for laser in self.snapshot.lasers:
            cosine = float(np.cos(laser.angle))
            sine = float(np.sin(laser.angle))
            start = (
                round((laser.x + cosine * laser.start_offset) * scale),
                round((laser.y + sine * laser.start_offset) * scale),
            )
            end = (
                round((laser.x + cosine * laser.end_offset) * scale),
                round((laser.y + sine * laser.end_offset) * scale),
            )
            width = max(1, round(laser.width * scale))
            pygame.draw.line(screen, (242, 93, 201), start, end, width)
        for bullet in self.snapshot.bullets:
            bullet_half_width = max(0.1, bullet.hitbox_width / 2.0)
            bullet_half_height = max(0.1, bullet.hitbox_height / 2.0)
            hitbox = pygame.Rect(
                round((bullet.x - bullet_half_width) * scale),
                round((bullet.y - bullet_half_height) * scale),
                max(1, round(bullet.hitbox_width * scale)),
                max(1, round(bullet.hitbox_height * scale)),
            )
            expanded = pygame.Rect(
                round((bullet.x - bullet_half_width - player_half_width) * scale),
                round((bullet.y - bullet_half_height - player_half_height) * scale),
                max(1, round((bullet_half_width + player_half_width) * 2.0 * scale)),
                max(1, round((bullet_half_height + player_half_height) * 2.0 * scale)),
            )
            pygame.draw.rect(screen, (102, 172, 255), hitbox)
            pygame.draw.rect(screen, (255, 142, 150), expanded, 1)

        player_hitbox = pygame.Rect(
            round((player.x - player_half_width) * scale),
            round((player.y - player_half_height) * scale),
            max(1, round(player.hitbox_width * scale)),
            max(1, round(player.hitbox_height * scale)),
        )
        pygame.draw.rect(screen, (255, 255, 255), player_hitbox)
        pygame.draw.rect(screen, (255, 78, 78), player_hitbox.inflate(6, 6), 1)

        if self.render_debug and self.last_observation is not None:
            self._draw_debug_panel(screen, int(TH06_PLAYFIELD_WIDTH * scale))
        pygame.display.flip()
        if self._render_clock is not None and self.render_fps is not None and self.render_fps > 0:
            self._render_clock.tick(self.render_fps)

    # Draw the three PCCM scales in their exact world-space windows.
    def _draw_pccm_projection(self, screen: Any, scale: float, screen_left: int = 0) -> None:
        import pygame

        if self.last_observation is None:
            return

        native_size = (TH06_PLAYFIELD_WIDTH, TH06_PLAYFIELD_HEIGHT)
        output_size = (
            int(TH06_PLAYFIELD_WIDTH * scale),
            int(TH06_PLAYFIELD_HEIGHT * scale),
        )
        if self._projection_overlay is None:
            self._projection_overlay = pygame.Surface(native_size, pygame.SRCALPHA)
        if self._projection_scaled_overlay is None or self._projection_scaled_overlay.get_size() != output_size:
            self._projection_scaled_overlay = pygame.Surface(output_size, pygame.SRCALPHA)
        overlay = self._projection_overlay
        overlay.fill((0, 0, 0, 0))
        layers = (
            ("blue_pccm", "_blue_window", (68, 139, 255)),
            ("yellow_pccm", "_yellow_window", (238, 198, 76)),
            ("red_pccm", "_red_window", (255, 82, 82)),
        )

        for map_key, window_key, border_color in layers:
            pccm = np.clip(np.asarray(self.last_observation[map_key], dtype=np.float32), 0.0, 1.0)
            window = np.asarray(self.last_observation[window_key], dtype=np.int32)
            x1, y1, x2, y2 = (int(value) for value in window)
            destination = pygame.Rect(
                x1,
                y1,
                max(1, x2 - x1),
                max(1, y2 - y1),
            )

            # The finer local scale replaces the coarser projection below it.
            visible_destination = destination.clip(overlay.get_rect())
            overlay.fill((0, 0, 0, 0), visible_destination)
            heatmap_size = (pccm.shape[1], pccm.shape[0])
            heatmap = self._projection_heatmaps.get(map_key)
            if heatmap is None or heatmap.get_size() != heatmap_size:
                heatmap = pygame.Surface(heatmap_size, pygame.SRCALPHA)
                self._projection_heatmaps[map_key] = heatmap
            rgb = pygame.surfarray.pixels3d(heatmap)
            alpha = pygame.surfarray.pixels_alpha(heatmap)
            transposed = pccm.T
            brightness = 0.50 + 0.50 * transposed
            for channel, color_value in enumerate(border_color):
                rgb[..., channel] = (float(color_value) * brightness).astype(np.uint8)
            alpha[...] = (48.0 + 132.0 * transposed).astype(np.uint8)
            del rgb, alpha
            scaled_heatmap = self._projection_scaled_heatmaps.get(map_key)
            if scaled_heatmap is None or scaled_heatmap.get_size() != destination.size:
                scaled_heatmap = pygame.Surface(destination.size, pygame.SRCALPHA)
                self._projection_scaled_heatmaps[map_key] = scaled_heatmap
            pygame.transform.scale(heatmap, destination.size, scaled_heatmap)
            overlay.blit(scaled_heatmap, destination.topleft)
            pygame.draw.rect(overlay, (*border_color, 210), destination, 2)

        playable_left, playable_top, playable_right, playable_bottom = TH06_PLAYABLE_BOUNDS
        movement_area = pygame.Rect(
            playable_left,
            playable_top,
            playable_right - playable_left,
            playable_bottom - playable_top,
        )
        pygame.draw.rect(overlay, (225, 230, 240, 130), movement_area, 1)
        pygame.transform.scale(overlay, output_size, self._projection_scaled_overlay)
        screen.blit(self._projection_scaled_overlay, (screen_left, 0))

    # Draw the local PCCM and compact live counters beside the playfield.
    def _draw_debug_panel(self, screen: Any, left: int) -> None:
        import pygame

        if self.last_observation is None:
            return
        panel = pygame.Rect(left, 0, 300, screen.get_height())
        pygame.draw.rect(screen, (25, 27, 32), panel)
        pccm = np.clip(np.asarray(self.last_observation["red_pccm"], dtype=np.float32), 0.0, 1.0)
        rgb = np.stack((255.0 * pccm, 75.0 * (1.0 - pccm), 120.0 * (1.0 - pccm)), axis=-1).astype(np.uint8)
        pccm_surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        pccm_surface = pygame.transform.scale(pccm_surface, (256, 256))
        screen.blit(pccm_surface, (left + 22, 48))
        pygame.draw.rect(screen, (180, 180, 185), (left + 22, 48, 256, 256), 1)
        font = pygame.font.Font(None, 24)
        lines = (
            "red PCCM",
            f"frame: {self.snapshot.game_frame}",
            f"bullets: {len(self.snapshot.bullets)}",
            f"lasers: {len(self.snapshot.lasers)}",
            f"enemies: {len(self.snapshot.enemies)}",
            f"reward: {self.episode_reward:+.3f}",
        )
        for index, line in enumerate(lines):
            y = 18 if index == 0 else 320 + (index - 1) * 28
            screen.blit(font.render(line, True, (235, 237, 242)), (left + 22, y))

    # Close the underlying original-game backend.
    def close(self) -> None:
        self.backend.close()
        if self._render_screen is not None:
            import pygame

            pygame.display.quit()
            self._render_screen = None
