# AntNavObstacle Specification

This document defines the DIVO-aligned Ant obstacle-navigation benchmark used as
a continuous-control validation task for automatic training-environment design.
It is a task specification, not an implementation note.

## 1. Goal

The goal is to instantiate the Ant experiment protocol used by DIVO as closely
as possible with publicly available Gymnasium Ant dynamics.

DIVO-style setting:

- Robot: MuJoCo Ant quadruped.
- Task: navigate from an initial region to a goal region.
- Action: 8D joint torques.
- Training environment: one randomly deployed small obstacle per episode.
- Test environments: held-out obstacle families B / M / U / D.
- Success: reach the goal without obstacle collision.

The resulting environment should support the same high-level paper claim as
Push-T:

> training-time obstacle distributions are automatically designed, while
> generalization is evaluated on unseen obstacle families.

## 2. Base Dynamics

Use Gymnasium `Ant-v5` only as the low-level dynamics model.

Official Ant-v5 facts:

- Action space: `Box(-1, 1, (8,), float32)`.
- Default observation: 105D.
- With `exclude_current_positions_from_observation=False`, torso `x/y` are
  included and the raw observation becomes 107D.
- Default reward is locomotion reward:
  `healthy_reward + forward_reward - ctrl_cost - contact_cost`.
- Default unhealthy termination is based on finite state values and torso
  height, with healthy `z` range `[0.2, 1.0]`.

For this benchmark, the default Ant reward is not used as the task reward. We
reuse Ant dynamics, action limits, health checks, and rendering, but replace the
task with goal navigation under obstacle constraints.

## 3. Task Geometry

Use a 2D ground-plane navigation task.

Default geometry:

- Start region: near `(-4.0, 0.0)`.
- Goal region: near `(4.0, 0.0)`.
- Workspace bounds: approximately `x in [-6, 6]`, `y in [-4, 4]`.
- Success threshold: torso-to-goal distance `<= 0.6`.
- Episode horizon: 500 to 700 environment steps.

The exact values can be tuned during sanity checks, but the benchmark protocol
must keep the same start/goal distribution across methods.

## 4. Obstacle Representation

The first implementation should use virtual 2D obstacles for reliable training
and evaluation.

Obstacle record:

```python
{
    "shape": "circle" | "box",
    "center": [x, y],
    "radius": r,              # for circle
    "half_size": [hx, hy],     # for box
    "angle": theta,            # optional for rotated box
    "active_after": 0.0,       # progress threshold for dynamic obstacles
}
```

Collision is computed from Ant torso `xy` against obstacle geometry. A collision
can terminate the episode or apply a large penalty; the benchmark success metric
must always count obstacle collision as failure.

Virtual obstacles are the default because they allow B/M/U/D protocols,
especially dynamic obstacles, without rebuilding MuJoCo XML mid-episode. Physical
MuJoCo geoms can be added later for visualization, but the benchmark definition
should not depend on them.

## 5. Observation and State

To match the DIVO Push-T structure, distinguish full observation from compact
state.

Full observation:

```text
o = compact_ant_state + relative_goal + obstacle_features
```

Compact state:

```text
s = compact_ant_state + relative_goal
```

Policy architecture should follow the DIVO pattern:

```text
z = encoder(o)
a = decoder(concat(s, z))
```

This means the policy is obstacle-aware through the latent encoder, while the
decoder receives an obstacle-free task state plus latent skill.

### Compact Ant State

DIVO reports a 39D Ant state. We should avoid using the raw 105D contact-force
observation as the paper state. The compact state should include:

- Torso planar position or position relative to start/goal.
- Torso height and orientation.
- Joint positions.
- Torso linear/angular velocity.
- Joint velocities.
- Relative goal vector.

Implementation detail:

- Use `Ant-v5` with `exclude_current_positions_from_observation=False` so torso
  `x/y` are directly available.
- Build the compact state by selecting position/velocity entries from the raw
  observation and appending relative-goal features.
- Keep the final dimension fixed in config. If the exact 39D layout is not
  recoverable from the public DIVO code, document the chosen compact layout and
  keep it consistent across all methods.

### Obstacle Features

For training with one obstacle:

```text
obstacle_features = [dx, dy, radius]
```

where `dx, dy` are obstacle center relative to torso or start-goal frame.

For benchmark families with multiple obstacle primitives, use a fixed maximum
number of obstacle slots and zero-pad unused slots:

```text
[dx_1, dy_1, size_1, active_1, ..., dx_K, dy_K, size_K, active_K]
```

The same observation dimension must be used during training and all evaluations.

## 6. Reward

Use a custom goal-navigation reward:

```text
r_t =
  w_progress * (d_{t-1} - d_t)
+ w_goal     * 1[success]
- w_ctrl     * ||a_t||^2
- w_collision * 1[collision]
- w_alive_fail * 1[unhealthy]
```

Recommended initial weights:

- `w_progress = 1.0`
- `w_goal = 50.0`
- `w_ctrl = 0.01`
- `w_collision = 50.0`
- `w_alive_fail = 10.0`

The reward can be tuned during the no-obstacle and single-obstacle sanity checks,
but the final training/evaluation protocol should be frozen before reporting
method comparisons.

## 7. Termination

Terminate when:

- Ant reaches the goal.
- Ant becomes unhealthy according to Gymnasium Ant health checks.
- Ant collides with an obstacle, if `terminate_on_collision=True`.

Truncate when:

- Episode horizon is reached.

For DIVO-style success evaluation:

```text
success = reached_goal and not collided and not unhealthy
```

## 8. Training Obstacle Distribution

Training uses a single small obstacle sampled between start and goal.

Sampling rule:

```text
c = start + alpha * (goal - start) + beta * n_perp
alpha ~ Uniform(alpha_min, alpha_max)
beta  ~ Uniform(-corridor_width, corridor_width)
r     = r_train
```

Recommended defaults:

- `alpha_min = 0.25`
- `alpha_max = 0.75`
- `corridor_width = 1.2`
- `r_train = 0.45`

Validity constraints:

- Obstacle must not overlap start region.
- Obstacle must not overlap goal region.
- Obstacle should lie in the start-goal corridor.
- If using multiple geometric primitives internally, they must not create an
  impossible fully blocked workspace.

This is the continuous Ant analogue of DIVO's between obstacle deployment.

## 9. Held-Out Evaluation Families

Evaluate the final policy without finetuning on the same four DIVO obstacle
families.

### Seen

Same distribution as training:

- One small obstacle.
- Between start and goal.
- Same radius range as training.

### B: Big Obstacle

One obstacle with larger size than training:

- `r_big > r_train`, e.g. `r_big = 0.75`.
- Same between corridor.

### M: Multiple Obstacles

Two small obstacles for Ant:

- `num_obstacles = 2`.
- Obstacles are separated along the start-goal corridor.
- Each obstacle has training-like size.

### U: U-Shape Obstacle

One non-convex obstacle family formed by multiple box primitives:

- A U-shaped blocker around the nominal straight path.
- Opening direction should vary across episodes.
- The Ant must detour around the non-convex region.

Represent this as several obstacle primitives but log/evaluate it as one U-shape
family.

### D: Dynamic Obstacle

One obstacle becomes active during the episode and blocks the original path.

Protocol:

- At reset, dynamic obstacle is inactive or placed out of the path.
- When progress ratio exceeds a threshold, activate a blocker in the
  start-goal corridor.
- Example threshold: `progress_ratio >= 0.45`.
- The obstacle should block the nominal straight path but leave a feasible
  detour.

This is easier to implement with virtual obstacles than with static MuJoCo XML.

## 10. Metrics

For every family report:

- Success rate.
- Collision rate.
- Unhealthy/fall rate.
- Timeout rate.
- Mean final distance to goal.
- Mean episode reward.
- Mean path efficiency, optional:
  `straight_line_distance / traveled_distance`.

Aggregate:

- Mean over B/M/U/D.
- Optionally also report Seen separately.

Outputs should mirror Push-T final evaluation:

```text
benchmark_summary.json
family-level videos
wandb scalar logs
optional obstacle layout figures
```

## 11. Training Algorithms

Primary paper-aligned algorithm:

- DIVO-style TD3 with `LatentDetPolicy`.

Sanity-check algorithm:

- SAC may be used only to verify that the environment reward and dynamics are
  learnable.

Final reported comparisons should use the same training algorithm across
methods. If TD3 is unstable and SAC is used in final Ant experiments, the paper
must state that Ant uses the same environment-design protocol but a separate
standard continuous-control learner.

## 12. Method Interfaces

Obstacle generator interface:

```python
def generate_obstacles(
    seed: int,
    start: list,
    goal: list,
    mode: str = "train",
) -> list:
    ...
```

Modes:

- `"train"`
- `"seen"`
- `"B"`
- `"M"`
- `"U"`
- `"D"`

For our automatic training-environment design method, the LLM should revise only
the training generator unless we explicitly study generated evaluation sets.
Held-out evaluation families must remain fixed and manually specified.

## 13. Sanity Checks Before Main Experiments

Run in this order:

1. No-obstacle AntGoal: confirm the learner reaches the goal.
2. Seen single-obstacle: confirm learning under one between obstacle.
3. Static evaluator: confirm B/M/U families run and log correctly.
4. Dynamic evaluator: confirm D activation and collision logic.
5. TD3 compatibility: confirm replay buffer, policy dimensions, and evaluator
   all use the fixed full observation dimension.
6. Video sanity: verify obstacle rendering overlays match collision geometry.

Only after these checks should we run seed-level method comparisons.

## 14. Paper Framing

Use this benchmark as the continuous locomotion counterpart to Push-T:

> In AntNavObstacle, we instantiate obstacle uncertainty as programmatic
> deployment of geometric constraints in a continuous locomotion task. Training
> uses single between-obstacle deployments, while zero-shot generalization is
> evaluated on held-out Big, Multiple, U-shape, and Dynamic obstacle families.

Do not claim that this is the exact unpublished DIVO Ant implementation. Claim
that it is a DIVO-aligned public re-instantiation based on Gymnasium Ant-v5.
