from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from DIVO.ant.generators import generate_obstacles
from DIVO.ant.obstacles import (
    ObstacleSpec,
    coerce_obstacles,
    collides,
    obstacle_feature_vector,
    progress_ratio,
)


class AntNavObstacleEnv(gym.Env):
    """DIVO-aligned Ant goal-navigation task with virtual obstacles."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        base_env_id: str = "Ant-v4",
        obstacle_mode: str = "train",
        obstacle: bool = True,
        start_xy: Sequence[float] = (-4.0, 0.0),
        goal_xy: Sequence[float] = (4.0, 0.0),
        start_noise: float = 0.25,
        goal_noise: float = 0.0,
        max_episode_steps: int = 700,
        success_threshold: float = 0.6,
        ant_body_radius: float = 0.35,
        terminate_on_collision: bool = True,
        terminate_on_unhealthy: bool = True,
        max_obstacles: int = 4,
        include_contact_forces: bool = False,
        render_mode: Optional[str] = None,
        render_style: str = "top_down",
        render_size: int = 512,
        render_world_scale: float = 1.2,
        fixed_obstacles: Optional[List[Dict[str, Any]]] = None,
        train_obstacle_radius: float = 0.45,
        train_alpha_min: float = 0.0,
        train_alpha_max: float = 1.0,
        train_lateral_ratio: float = 0.15,
        train_start_clearance: float = 0.20,
        train_goal_clearance: float = 0.25,
        train_corridor_width: float = 1.2,
        reward_goal_velocity: float = 1.0,
        reward_progress: float = 0.0,
        reward_goal: float = 50.0,
        reward_ctrl: float = 0.01,
        reward_collision: float = 50.0,
        reward_unhealthy: float = 10.0,
        reward_clearance: float = 0.0,
        clearance_margin: float = 0.4,
        normalize_task_obs: bool = False,
        **base_env_kwargs,
    ):
        super().__init__()
        self.base_env_id = base_env_id
        self.obstacle_mode = obstacle_mode
        self.obstacle_enabled = bool(obstacle)
        self.start_xy_base = np.asarray(start_xy, dtype=np.float64)
        self.goal_xy_base = np.asarray(goal_xy, dtype=np.float64)
        self.start_noise = float(start_noise)
        self.goal_noise = float(goal_noise)
        self.max_episode_steps = int(max_episode_steps)
        self.success_threshold = float(success_threshold)
        self.ant_body_radius = float(ant_body_radius)
        self.terminate_on_collision = bool(terminate_on_collision)
        self.terminate_on_unhealthy = bool(terminate_on_unhealthy)
        self.max_obstacles = int(max_obstacles)
        self.include_contact_forces = bool(include_contact_forces)
        self.render_mode = render_mode
        self.render_style = str(render_style)
        self.render_size = int(render_size)
        self.render_world_scale = float(render_world_scale)
        self.fixed_obstacles = coerce_obstacles(fixed_obstacles or [])
        self.train_obstacle_radius = float(train_obstacle_radius)
        self.train_alpha_min = float(train_alpha_min)
        self.train_alpha_max = float(train_alpha_max)
        self.train_lateral_ratio = float(train_lateral_ratio)
        self.train_start_clearance = float(train_start_clearance)
        self.train_goal_clearance = float(train_goal_clearance)
        self.train_corridor_width = float(train_corridor_width)
        self.reward_goal_velocity = float(reward_goal_velocity)
        self.reward_progress = float(reward_progress)
        self.reward_goal = float(reward_goal)
        self.reward_ctrl = float(reward_ctrl)
        self.reward_collision = float(reward_collision)
        self.reward_unhealthy = float(reward_unhealthy)
        self.reward_clearance = float(reward_clearance)
        self.clearance_margin = float(clearance_margin)
        self.normalize_task_obs = bool(normalize_task_obs)
        self.base_env_kwargs = dict(base_env_kwargs)

        self.base_env = self._make_base_env(render_mode=render_mode)
        self.action_space = self.base_env.action_space

        self.compact_state_dim = 39
        self.relative_goal_dim = 2
        self.state_dim = self.compact_state_dim + self.relative_goal_dim
        self.obstacle_feature_dim = self.max_obstacles * 5
        self.obs_dim = [self.state_dim + self.obstacle_feature_dim]
        self.action_dim = [int(np.prod(self.action_space.shape))]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, self.obs_dim[0]),
            dtype=np.float32,
        )

        self.start_xy = self.start_xy_base.copy()
        self.goal_xy = self.goal_xy_base.copy()
        self.obstacles: List[ObstacleSpec] = []
        self.step_count = 0
        self.prev_distance = 0.0
        self.last_xy = self.start_xy.copy()
        self.trajectory: List[np.ndarray] = []
        self.collided = False
        self.collision_obstacle: Optional[ObstacleSpec] = None
        self._rng = np.random.default_rng()
        self._last_raw_obs: Optional[np.ndarray] = None
        self._ant_sprite_renderer = None

    def _make_base_env(self, render_mode: Optional[str]):
        kwargs = dict(self.base_env_kwargs)
        kwargs.setdefault("exclude_current_positions_from_observation", False)
        kwargs.setdefault("terminate_when_unhealthy", True)
        if self.include_contact_forces or not str(self.base_env_id).endswith("-v5"):
            kwargs.setdefault("use_contact_forces", self.include_contact_forces)
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        try:
            return gym.make(self.base_env_id, **kwargs)
        except gym.error.Error:
            if self.base_env_id == "Ant-v5":
                self.base_env_id = "Ant-v4"
                return gym.make(self.base_env_id, **kwargs)
            raise

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        return [seed]

    def get_info(self):
        return self.action_dim, self.obs_dim

    def obs2state(self, obs):
        return obs[:, : self.state_dim]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        options = dict(options or {})

        raw_obs, raw_info = self.base_env.reset(seed=seed)
        self.start_xy = self._sample_xy(self.start_xy_base, self.start_noise)
        self.goal_xy = self._sample_xy(self.goal_xy_base, self.goal_noise)
        if "start_xy" in options:
            self.start_xy = np.asarray(options["start_xy"], dtype=np.float64)
        if "goal_xy" in options:
            self.goal_xy = np.asarray(options["goal_xy"], dtype=np.float64)

        self._set_ant_xy(self.start_xy)
        raw_obs = self._get_raw_obs()
        self._last_raw_obs = raw_obs
        self.last_xy = self._torso_xy(raw_obs)
        self.trajectory = [self.last_xy.copy()]
        self.prev_distance = self._goal_distance(self.last_xy)
        self.step_count = 0
        self.collided = False
        self.collision_obstacle = None

        if "obstacles" in options:
            self.obstacles = coerce_obstacles(options["obstacles"])
        elif self.fixed_obstacles:
            self.obstacles = list(self.fixed_obstacles)
        elif self.obstacle_enabled:
            self.obstacles = generate_obstacles(
                seed=None if seed is None else int(seed),
                start=self.start_xy,
                goal=self.goal_xy,
                mode=self.obstacle_mode,
                train_radius=self.train_obstacle_radius,
                train_alpha_min=self.train_alpha_min,
                train_alpha_max=self.train_alpha_max,
                train_lateral_ratio=self.train_lateral_ratio,
                train_body_radius=self.ant_body_radius,
                train_start_clearance=self.train_start_clearance,
                train_goal_clearance=self.train_goal_clearance,
            )
        else:
            self.obstacles = []

        obs = self._build_obs(raw_obs)
        info = self._build_info(raw_info, raw_obs, reward_dict={})
        return obs, info

    def step(self, action):
        raw_obs, base_reward, base_terminated, base_truncated, raw_info = self.base_env.step(action)
        self._last_raw_obs = raw_obs
        xy = self._torso_xy(raw_obs)
        distance = self._goal_distance(xy)
        prog = progress_ratio(self.start_xy, self.goal_xy, xy)
        hit, hit_obstacle, margin = collides(
            xy,
            self.obstacles,
            progress_ratio=prog,
            body_radius=self.ant_body_radius,
        )
        self.collided = self.collided or hit
        if hit and self.collision_obstacle is None:
            self.collision_obstacle = hit_obstacle

        success = distance <= self.success_threshold and not self.collided
        unhealthy = self._is_unhealthy()
        progress_reward = self.prev_distance - distance
        dt = float(getattr(self.base_env.unwrapped, "dt", 0.05))
        displacement = xy - self.last_xy
        velocity = displacement / max(dt, 1e-8)
        goal_direction = self._goal_direction(self.last_xy)
        raw_goal_velocity = float(np.dot(velocity, goal_direction))
        goal_velocity_reward = self.reward_goal_velocity * raw_goal_velocity
        weighted_progress_reward = self.reward_progress * progress_reward
        raw_ctrl_cost = float(np.sum(np.square(np.asarray(action, dtype=np.float64))))
        ctrl_cost = self.reward_ctrl * raw_ctrl_cost
        goal_bonus = self.reward_goal * float(success)
        collision_penalty = self.reward_collision * float(hit)
        unhealthy_penalty = self.reward_unhealthy * float(unhealthy)
        clearance_gap = max(0.0, self.clearance_margin - float(margin))
        near_obstacle_penalty = self.reward_clearance * clearance_gap * clearance_gap
        reward = (
            goal_velocity_reward
            + weighted_progress_reward
            + goal_bonus
            - ctrl_cost
            - collision_penalty
            - unhealthy_penalty
            - near_obstacle_penalty
        )
        reward_dict = {
            "task_reward": float(reward),
            "base_reward": float(base_reward),
            "progress_reward": float(progress_reward),
            "weighted_progress_reward": float(weighted_progress_reward),
            "goal_velocity_reward": float(goal_velocity_reward),
            "raw_goal_velocity": float(raw_goal_velocity),
            "goal_bonus": float(goal_bonus),
            "ctrl_cost": float(ctrl_cost),
            "raw_ctrl_cost": float(raw_ctrl_cost),
            "collision_penalty": float(collision_penalty),
            "unhealthy_penalty": float(unhealthy_penalty),
            "near_obstacle_penalty": float(near_obstacle_penalty),
            "clearance_gap": float(clearance_gap),
            "goal_distance": float(distance),
            "progress_ratio": float(prog),
            "collision_margin": float(margin),
        }

        self.prev_distance = distance
        self.last_xy = xy.copy()
        self.trajectory.append(self.last_xy.copy())
        self.step_count += 1
        terminated = bool(
            success
            or (unhealthy and self.terminate_on_unhealthy)
            or (hit and self.terminate_on_collision)
        )
        truncated = bool(base_truncated or self.step_count >= self.max_episode_steps)
        obs = self._build_obs(raw_obs)
        info = self._build_info(raw_info, raw_obs, reward_dict=reward_dict)
        info["success"] = bool(success)
        info["collision"] = bool(hit)
        info["collided"] = bool(self.collided)
        info["unhealthy"] = bool(unhealthy)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_style in ("top_down", "topdown", "2d"):
            return self._render_top_down()
        return self.base_env.render()

    def close(self):
        if self._ant_sprite_renderer is not None:
            self._ant_sprite_renderer.close()
            self._ant_sprite_renderer = None
        self.base_env.close()

    def _sample_xy(self, base: np.ndarray, noise: float) -> np.ndarray:
        if noise <= 0:
            return base.copy()
        return base + self._rng.uniform(-noise, noise, size=2)

    def _get_raw_obs(self) -> np.ndarray:
        if hasattr(self.base_env.unwrapped, "_get_obs"):
            return np.asarray(self.base_env.unwrapped._get_obs(), dtype=np.float64)
        if self._last_raw_obs is None:
            raise RuntimeError("Cannot obtain Ant raw observation before reset.")
        return np.asarray(self._last_raw_obs, dtype=np.float64)

    def _set_ant_xy(self, xy: Sequence[float]):
        unwrapped = self.base_env.unwrapped
        qpos = np.array(unwrapped.data.qpos, copy=True)
        qvel = np.array(unwrapped.data.qvel, copy=True)
        qpos[:2] = np.asarray(xy, dtype=np.float64)[:2]
        unwrapped.set_state(qpos, qvel)

    def _torso_xy(self, raw_obs: np.ndarray) -> np.ndarray:
        return np.asarray(raw_obs[:2], dtype=np.float64)

    def _goal_distance(self, xy: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - self.goal_xy))

    def _goal_direction(self, xy: Sequence[float]) -> np.ndarray:
        direction = self.goal_xy - np.asarray(xy, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            return np.zeros(2, dtype=np.float64)
        return direction / norm

    def _render_top_down(self) -> np.ndarray:
        size = max(int(self.render_size), 128)
        pad = int(0.08 * size)
        world_min, world_max = self._render_world_bounds()
        span = np.maximum(world_max - world_min, 1e-6)
        drawable = max(size - 2 * pad, 1)

        def to_px(xy: Sequence[float]) -> Tuple[float, float]:
            point = np.asarray(xy, dtype=np.float64)
            x = pad + (point[0] - world_min[0]) / span[0] * drawable
            y = size - pad - (point[1] - world_min[1]) / span[1] * drawable
            return float(x), float(y)

        def scale_len(value: float) -> float:
            return float(value) / float(max(span[0], span[1])) * drawable

        img = Image.new("RGB", (size, size), (248, 249, 247))
        draw = ImageDraw.Draw(img, "RGBA")
        self._draw_checkerboard(draw, size=size, pad=pad, world_min=world_min, world_max=world_max)

        # Task corridor and goal line.
        sx, sy = to_px(self.start_xy)
        gx, gy = to_px(self.goal_xy)
        draw.line((sx, sy, gx, gy), fill=(80, 80, 80, 150), width=max(2, size // 180))

        # Obstacles, body-collision envelope, and optional clearance envelope.
        for obstacle in self.obstacles:
            self._draw_obstacle(draw, obstacle, to_px, scale_len)

        # Trajectory behind the current Ant position.
        if len(self.trajectory) > 1:
            pts = [to_px(point) for point in self.trajectory[-500:]]
            draw.line(pts, fill=(33, 105, 172, 210), width=max(2, size // 140), joint="curve")

        # Goal and current Ant torso. The initial Ant pose itself serves as the start marker.
        self._draw_marker(draw, (gx, gy), radius=max(10, size // 28), fill=(38, 200, 70, 255), outline=(18, 115, 40, 255))
        self._draw_mujoco_ant_sprite(img, draw, to_px, scale_len)

        self._draw_label(draw, "G", (gx, gy), fill=(255, 255, 255, 255))

        return np.asarray(img, dtype=np.uint8)

    def _render_world_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        points = [self.start_xy, self.goal_xy]
        radii = [
            self.ant_body_radius
            + max(self.clearance_margin, 0.0)
            + max(self.train_corridor_width, 0.0) * 0.5
            + 0.3
        ]
        for obstacle in self.obstacles:
            points.append(np.asarray(obstacle.center, dtype=np.float64))
            if obstacle.shape == "circle":
                radii.append(float(obstacle.radius or 0.0) + self.ant_body_radius + max(self.clearance_margin, 0.0))
            elif obstacle.shape == "box" and obstacle.half_size is not None:
                radii.append(float(np.linalg.norm(obstacle.half_size)) + self.ant_body_radius + max(self.clearance_margin, 0.0))
        arr = np.asarray(points, dtype=np.float64)
        margin = max(0.8, max(radii) if radii else 0.8)
        world_min = arr.min(axis=0) - margin
        world_max = arr.max(axis=0) + margin
        center = 0.5 * (world_min + world_max)
        half = 0.5 * max(float(world_max[0] - world_min[0]), float(world_max[1] - world_min[1]), 1.0)
        half *= max(self.render_world_scale, 1e-6)
        return center - half, center + half

    def _draw_checkerboard(self, draw: ImageDraw.ImageDraw, size: int, pad: int, world_min: np.ndarray, world_max: np.ndarray):
        span = max(float(world_max[0] - world_min[0]), float(world_max[1] - world_min[1]), 1e-6)
        cell = max(int((size - 2 * pad) / max(span, 1.0) * 0.5), 18)
        colors = ((236, 241, 235, 255), (142, 159, 142, 255))
        for y in range(pad, size - pad, cell):
            for x in range(pad, size - pad, cell):
                idx = ((x - pad) // cell + (y - pad) // cell) % 2
                draw.rectangle((x, y, min(x + cell, size - pad), min(y + cell, size - pad)), fill=colors[idx])
        draw.rectangle((pad, pad, size - pad, size - pad), outline=(115, 135, 115, 180), width=max(1, size // 220))

    def _draw_obstacle(self, draw: ImageDraw.ImageDraw, obstacle: ObstacleSpec, to_px, scale_len):
        cx, cy = to_px(obstacle.center)
        if obstacle.shape == "circle":
            radius = float(obstacle.radius or 0.0)
            clear_r = scale_len(radius + self.ant_body_radius + max(self.clearance_margin, 0.0))
            collide_r = scale_len(radius + self.ant_body_radius)
            obj_r = scale_len(radius)
            if clear_r > 1:
                draw.ellipse((cx - clear_r, cy - clear_r, cx + clear_r, cy + clear_r), fill=(255, 0, 0, 32), outline=(255, 0, 0, 120), width=2)
            if collide_r > 1:
                draw.ellipse((cx - collide_r, cy - collide_r, cx + collide_r, cy + collide_r), outline=(255, 0, 0, 190), width=max(2, self.render_size // 160))
            draw.ellipse((cx - obj_r, cy - obj_r, cx + obj_r, cy + obj_r), fill=(255, 0, 0, 235), outline=(150, 0, 0, 255), width=max(1, self.render_size // 220))
            return

        if obstacle.shape == "box" and obstacle.half_size is not None:
            hx, hy = obstacle.half_size
            half_x = scale_len(hx)
            half_y = scale_len(hy)
            draw.rounded_rectangle((cx - half_x, cy - half_y, cx + half_x, cy + half_y), radius=3, fill=(255, 0, 0, 235), outline=(150, 0, 0, 255), width=2)

    def _draw_marker(self, draw: ImageDraw.ImageDraw, center: Tuple[float, float], radius: float, fill, outline):
        x, y = center
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)

    def _draw_mujoco_ant_sprite(self, img: Image.Image, draw: ImageDraw.ImageDraw, to_px, scale_len):
        try:
            sprite = self._render_mujoco_ant_sprite()
            target_box = self._ant_sprite_target_box(to_px, scale_len)
            if sprite is None or target_box is None:
                self._draw_mujoco_ant_projection(draw, to_px, scale_len)
                return
            x0, y0, x1, y1 = target_box
            width = max(1, int(round(x1 - x0)))
            height = max(1, int(round(y1 - y0)))
            sprite = sprite.resize((width, height), Image.Resampling.LANCZOS)
            img.paste(sprite, (int(round(x0)), int(round(y0))), sprite)
        except Exception:
            self._draw_mujoco_ant_projection(draw, to_px, scale_len)

    def _render_mujoco_ant_sprite(self) -> Optional[Image.Image]:
        import mujoco

        model = self.base_env.unwrapped.model
        data = self.base_env.unwrapped.data
        if self._ant_sprite_renderer is None:
            self._ant_sprite_renderer = mujoco.Renderer(model, height=384, width=384)

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [float(self.last_xy[0]), float(self.last_xy[1]), 0.35]
        camera.distance = 2.4
        camera.azimuth = 90
        camera.elevation = -75

        option = mujoco.MjvOption()
        renderer = self._ant_sprite_renderer
        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera=camera, scene_option=option)
        rgb = renderer.render()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera, scene_option=option)
        segmentation = renderer.render()
        renderer.disable_segmentation_rendering()

        geom_ids = segmentation[..., 0]
        mask = geom_ids > 0
        if not np.any(mask):
            return None
        ys, xs = np.where(mask)
        pad = 8
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, rgb.shape[1])
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, rgb.shape[0])
        alpha = np.zeros(mask.shape, dtype=np.uint8)
        alpha[mask] = 255
        rgba = np.dstack([rgb, alpha])
        return self._paper_style_ant_sprite(Image.fromarray(rgba[y0:y1, x0:x1], mode="RGBA"))

    def _paper_style_ant_sprite(self, sprite: Image.Image) -> Image.Image:
        sprite = sprite.convert("RGBA")
        rgb = sprite.convert("RGB")
        rgb = ImageEnhance.Color(rgb).enhance(0.92)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.18)
        rgb = ImageEnhance.Brightness(rgb).enhance(0.88)
        alpha = sprite.getchannel("A").filter(ImageFilter.GaussianBlur(radius=0.35))

        dark = Image.new("RGBA", sprite.size, (28, 24, 20, 0))
        outline_alpha = alpha.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=0.85))
        outline_alpha = outline_alpha.point(lambda value: int(value * 0.75))
        dark.putalpha(outline_alpha)

        body = Image.merge("RGBA", (*rgb.split(), alpha))
        composed = Image.alpha_composite(dark, body)
        shadow = Image.new("RGBA", (composed.width + 8, composed.height + 8), (0, 0, 0, 0))
        shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=2.0)).point(lambda value: int(value * 0.22))
        shadow_layer = Image.new("RGBA", sprite.size, (20, 18, 16, 0))
        shadow_layer.putalpha(shadow_alpha)
        shadow.paste(shadow_layer, (5, 5), shadow_layer)
        shadow.paste(composed, (0, 0), composed)
        return shadow

    def _ant_sprite_target_box(self, to_px, scale_len) -> Optional[Tuple[float, float, float, float]]:
        bounds = self._ant_geom_xy_bounds()
        if bounds is None:
            return None
        world_min, world_max = bounds
        x0, y0 = to_px([world_min[0], world_max[1]])
        x1, y1 = to_px([world_max[0], world_min[1]])
        pad = max(scale_len(0.08), 3.0)
        x0 -= pad
        y0 -= pad
        x1 += pad
        y1 += pad
        center_x = 0.5 * (x0 + x1)
        center_y = 0.5 * (y0 + y1)
        scale = 0.78
        half_w = 0.5 * (x1 - x0) * scale
        half_h = 0.5 * (y1 - y0) * scale
        return center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h

    def _ant_geom_xy_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        model = self.base_env.unwrapped.model
        data = self.base_env.unwrapped.data
        points = []
        for geom_id in range(model.ngeom):
            name = str(model.geom(geom_id).name)
            if name == "floor":
                continue
            geom_type = int(model.geom_type[geom_id])
            size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            xy = np.asarray(data.geom_xpos[geom_id][:2], dtype=np.float64)
            radius = float(size[0])
            if geom_type == 2:
                points.append(xy - radius)
                points.append(xy + radius)
                continue
            if geom_type == 3:
                xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
                axis_xy = xmat[:2, 2]
                axis_norm = float(np.linalg.norm(axis_xy))
                if axis_norm <= 1e-8:
                    points.append(xy - radius)
                    points.append(xy + radius)
                    continue
                axis_xy = axis_xy / axis_norm
                half_len = float(size[1])
                p0 = xy - axis_xy * half_len
                p1 = xy + axis_xy * half_len
                points.extend([p0 - radius, p0 + radius, p1 - radius, p1 + radius])
        if not points:
            return None
        arr = np.asarray(points, dtype=np.float64)
        return arr.min(axis=0), arr.max(axis=0)

    def _draw_mujoco_ant_projection(self, draw: ImageDraw.ImageDraw, to_px, scale_len):
        model = self.base_env.unwrapped.model
        data = self.base_env.unwrapped.data
        geom_order = list(range(model.ngeom))
        geom_order.sort(key=lambda idx: 1 if str(model.geom(idx).name) == "torso_geom" else 0)
        for geom_id in geom_order:
            name = str(model.geom(geom_id).name)
            if name == "floor":
                continue
            geom_type = int(model.geom_type[geom_id])
            size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float64)
            fill = tuple(int(np.clip(channel, 0.0, 1.0) * 255) for channel in rgba)
            outline = (84, 60, 38, 255)
            xy = np.asarray(data.geom_xpos[geom_id][:2], dtype=np.float64)
            cx, cy = to_px(xy)

            if geom_type == 2:  # sphere
                radius = max(scale_len(float(size[0])), 3.0)
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=fill,
                    outline=outline,
                    width=max(1, self.render_size // 220),
                )
                highlight = radius * 0.45
                draw.ellipse(
                    (cx - highlight, cy - highlight, cx + highlight, cy + highlight),
                    fill=(255, 218, 154, 190),
                    outline=None,
                )
                continue

            if geom_type == 3:  # capsule
                radius = max(scale_len(float(size[0])), 2.0)
                half_len = scale_len(float(size[1]))
                xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
                axis_xy = xmat[:2, 2]
                axis_norm = float(np.linalg.norm(axis_xy))
                if axis_norm <= 1e-8 or half_len <= 1e-8:
                    draw.ellipse(
                        (cx - radius, cy - radius, cx + radius, cy + radius),
                        fill=fill,
                        outline=outline,
                        width=1,
                    )
                    continue
                axis_xy = axis_xy / axis_norm
                p0_world = xy - axis_xy * float(size[1])
                p1_world = xy + axis_xy * float(size[1])
                p0 = to_px(p0_world)
                p1 = to_px(p1_world)
                width = max(2, int(round(2.0 * radius)))
                draw.line((*p0, *p1), fill=outline, width=width + 2)
                draw.line((*p0, *p1), fill=fill, width=width)
                for px, py in (p0, p1):
                    draw.ellipse(
                        (px - radius, py - radius, px + radius, py + radius),
                        fill=fill,
                        outline=outline,
                        width=1,
                    )

    def _draw_label(self, draw: ImageDraw.ImageDraw, text: str, center: Tuple[float, float], fill):
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, self.render_size // 30))
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        x, y = center
        draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2), text, fill=fill, font=font)

    def _task_scale(self) -> float:
        return max(float(np.linalg.norm(self.goal_xy - self.start_xy)), 1e-8)

    def _is_unhealthy(self) -> bool:
        unwrapped = self.base_env.unwrapped
        if hasattr(unwrapped, "is_healthy"):
            return not bool(unwrapped.is_healthy)
        return False

    def _compact_ant_state(self, raw_obs: np.ndarray) -> np.ndarray:
        compact = np.zeros(self.compact_state_dim, dtype=np.float32)
        take = min(self.compact_state_dim, raw_obs.shape[0])
        compact[:take] = raw_obs[:take].astype(np.float32)
        return compact

    def _build_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        xy = self._torso_xy(raw_obs)
        prog = progress_ratio(self.start_xy, self.goal_xy, xy)
        scale = self._task_scale() if self.normalize_task_obs else 1.0
        compact = self._compact_ant_state(raw_obs)
        relative_goal = ((self.goal_xy - xy) / scale).astype(np.float32)
        obstacle_features = obstacle_feature_vector(
            xy,
            self.obstacles,
            progress_ratio=prog,
            max_obstacles=self.max_obstacles,
            scale=scale,
        )
        flat = np.concatenate([compact, relative_goal, obstacle_features], axis=0).astype(np.float32)
        return flat.reshape(1, -1)

    def _build_info(self, raw_info: Dict[str, Any], raw_obs: np.ndarray, reward_dict: Dict[str, float]):
        xy = self._torso_xy(raw_obs)
        prog = progress_ratio(self.start_xy, self.goal_xy, xy)
        distance = self._goal_distance(xy)
        active_obstacles = [obs for obs in self.obstacles if prog >= obs.active_after]
        info = dict(raw_info)
        info.update(
            {
                "xy_position": xy.copy(),
                "x_position": float(xy[0]),
                "y_position": float(xy[1]),
                "start_xy": self.start_xy.copy(),
                "goal_xy": self.goal_xy.copy(),
                "goal_distance": float(distance),
                "progress_ratio": float(prog),
                "obstacles": [obs.to_dict() for obs in self.obstacles],
                "active_obstacle_count": len(active_obstacles),
                "reward_dict": reward_dict,
            }
        )
        return info
