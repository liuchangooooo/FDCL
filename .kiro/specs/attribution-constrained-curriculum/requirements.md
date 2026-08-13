# Requirements Document: Attribution-Constrained Curriculum

## Introduction

The DIVO + ACGS curriculum on Push-T currently follows the loop:

```
attribution evidence → user_prompt.txt → LLM (free-form code) → sanity_check → accepted as new generator → train
```

In the diagnostic run `data/outputs/2026.05.28/21.58.13_td3_pusht_llm_curriculum/`,
attribution correctly identified high-lift, high-failure-rate cells
(e.g. `alpha=050_to_075|beta_abs=near_side|blockage=medium`, lift 7.04,
failure_rate 0.873). The LLM read this evidence in `user_prompt.txt`. Across 8
evolutions, however, the resulting generators concentrated 50%-90% of obstacle
samples on `far_side|low` cells with failure_rate < 0.15, and only 0.3%-1.7% on
the highest-lift `near_side` cells. Global success rate stayed at
0.85-0.94 ("too_easy") while the policy still failed on hard cells
~87% of the time.

Two coupled mechanisms produce this outcome:

1. **Interpretation gap.** The LLM is the only channel that turns evidence
   into a sampling distribution. It can interpret "high-lift = dangerous,
   avoid" rather than "high-lift = high training value, oversample".
2. **Intent-vs-realization gap.** Even when the LLM's intent is to
   oversample near_side cells, the procedural rejection chain in
   `generate_obstacles` (`is_safe()`, `start_clear`, `target_clear`,
   multi-layer fallbacks) silently redirects mass to far_side, and the LLM
   gets no feedback on its own generator's empirical coverage.

This spec captures the user-facing requirements for an
**Attribution-Constrained Curriculum** in which a deterministic Python-side
controller owns the cell-level sampling distribution, the LLM is relegated
to a proposer role, and every accepted generator is validated against
declared intent before training. The requirements deliberately surface
several candidate design directions (structured-output proposals,
"what" vs "how" decoupling, population mixture, adversarial framing) as
opt-in modes; selection between them is left to `design.md`.

## Glossary

- **TD3CurriculumWorkspace**: existing workspace at
  `DIVO/workspace/rl_workspace/td3_curriculum_workspace.py` that owns the
  evolve loop.
- **ACGS_API**: existing LLM facade at `DIVO/gpt/acgs_api.py`.
- **PromptBuilder**: existing prompt assembler at
  `DIVO/gpt/prompt_builder.py`.
- **Generator**: Python callable matching the contract
  `generate_obstacles(tblock_pose: np.ndarray, num_obstacles: int) -> list[dict]`
  where each dict carries `x`, `y`, and optionally `purpose`.
- **StrategyExecutor**: existing sandbox at
  `DIVO/env/pusht/llm_topology_generator.py` that exposes `is_safe(...)`,
  `decode_obstacle(...)`, `encode_obstacle(...)` to LLM-supplied code.
- **Cell**: 3-tuple `(alpha_bin, beta_abs_bin, blockage_bin)` produced by
  `cell_key(...)` in `DIVO/curriculum/attribution.py`. `cell_id` is the
  string form `alpha=...|beta_abs=...|blockage=...`.
- **AttributionResult**: output of
  `DIVO.curriculum.attribution.compute_attribution`.
- **HighLiftCells**: subset of `AttributionResult.cells` with
  `failure_lift >= controller.min_lift` and
  `total_count >= AttributionConfig.min_support`.
- **MassBudget**: per-cell sampling distribution (per-cell target
  probability and per-cell minimum probability) for the next training
  round, owned by the CurriculumController.
- **CurriculumController**: new Python-side module that derives MassBudget
  from AttributionResult, validates Generator candidates against MassBudget,
  and decides accept/reject.
- **GeneratorProposal**: artifact proposed by the LLM in response to an
  evolve prompt; either free-form Python code or a structured JSON spec.
- **GeneratorCompiler**: deterministic Python module that compiles a
  structured GeneratorProposal into a Generator.
- **Sampler**: cell-conditioned realization layer that draws a Cell from
  MassBudget, then draws an `(x, y)` from the corresponding cell envelope.
- **DryRunValidator**: offline rollout of a Generator (no policy training)
  over a configurable number of seeded calls, encoded into per-cell
  CoverageReport via `pusht_encode_layout` from
  `DIVO/curriculum/adapters/pusht_adapter.py`.
- **CoverageReport**: aggregate of DryRun results; per-cell empirical
  share, per-cell `(target, realized, deviation)`, and per-layout
  feasibility outcomes when the FeasibilityOracle is enabled.
- **FeasibilityOracle**: deterministic solvability check for a layout
  given start pose, target pose, and obstacle list.
- **DifficultySignal**: per-cell or lift-weighted summary of policy
  performance restricted to HighLiftCells, replacing the global
  `success_rate` trigger in `_difficulty_reason`.
- **EvolveRound**: one iteration of the curriculum loop; produces
  artifacts under `<output_dir>/controller/evolve_NNN/`.

## Requirements

### Requirement 1: Deterministic ownership of the cell-level mass distribution

**User Story:** As a trainer, I want a Python-side CurriculumController to
own the cell-level mass distribution that the next training round will
realize, so that the actual training distribution is no longer determined
by the LLM's interpretation of evidence.

#### Acceptance Criteria

1. THE CurriculumController SHALL maintain a MassBudget (per-cell target
   probability and per-cell minimum probability) that is derived before the
   LLM is invoked for each EvolveRound.
2. WHEN an AttributionResult is available for the current batch, THE
   CurriculumController SHALL derive the next-round MassBudget from that
   AttributionResult without LLM input.
3. THE CurriculumController SHALL persist the MassBudget for each
   EvolveRound to `<output_dir>/controller/evolve_NNN/mass_budget.json`.
4. THE MassBudget SHALL satisfy: the sum of per-cell target probabilities
   equals 1.0 within an absolute tolerance of 1.0e-6.
5. WHEN the same AttributionResult and the same controller configuration
   are provided, THE CurriculumController SHALL produce a byte-identical
   `mass_budget.json` across runs.

### Requirement 2: Hard mass-budget constraints derived from attribution

**User Story:** As a trainer, I want top-lift cells to receive a guaranteed
minimum sampling share, so that high-lift evidence translates into actual
training exposure rather than into prompt text the LLM may misinterpret.

#### Acceptance Criteria

1. THE CurriculumController SHALL select the HighLiftCells subset using
   `failure_lift >= controller.min_lift` (default 1.5) and
   `total_count >= AttributionConfig.min_support`.
2. THE CurriculumController SHALL assign each HighLiftCell a per-cell
   minimum probability `p_min_high_lift` configurable via Hydra under
   `curriculum.controller.p_min_high_lift`.
3. THE CurriculumController SHALL ensure that the sum of per-cell minimum
   probabilities across HighLiftCells does not exceed
   `controller.max_high_lift_total_mass` (default 0.6).
4. WHERE no Cell satisfies the HighLiftCells selection, THE
   CurriculumController SHALL fall back to a uniform MassBudget over Cells
   with `total_count >= AttributionConfig.min_support` and SHALL emit a
   structured warning record to `train.log` with key
   `controller.no_high_lift_cells`.
5. IF a Generator candidate cannot realize the MassBudget within
   `controller.mass_tolerance` after DryRun, THEN THE CurriculumController
   SHALL reject that candidate and SHALL not advance `evolve_count`.

### Requirement 3: Closed-loop validation against declared intent (DryRun)

**User Story:** As a trainer, I want every Generator candidate to be
dry-run before it is allowed to drive training, so that the controller's
declared intent becomes realized empirical coverage rather than being
silently leaked to far_side cells by procedural fallbacks.

#### Acceptance Criteria

1. WHEN a Generator candidate is loaded, THE DryRunValidator SHALL invoke
   that Generator for `controller.dry_run_calls` calls (default 1000)
   using a seeded RNG and the `tblock_pose` distribution defined by
   `TD3CurriculumWorkspace._make_llm_scene`.
2. THE DryRunValidator SHALL encode each generated layout via
   `pusht_encode_layout` and SHALL aggregate per-cell counts into a
   CoverageReport.
3. THE DryRunValidator SHALL compare CoverageReport against MassBudget
   using total variation distance and per-cell absolute deviation.
4. IF the empirical share for any HighLiftCell is below
   `MassBudget[cell] - controller.mass_tolerance` (default 0.05), THEN
   THE DryRunValidator SHALL reject the Generator candidate.
5. THE DryRunValidator SHALL persist CoverageReport, MassBudget, and
   per-cell deviation to
   `<output_dir>/controller/evolve_NNN/coverage_report.json` and
   `<output_dir>/controller/evolve_NNN/per_cell_deviation.csv` for every
   accepted and rejected candidate.
6. WHEN given the same Generator code, the same dry-run seed, and the
   same `controller.dry_run_calls`, THE DryRunValidator SHALL produce a
   byte-identical CoverageReport across runs.

### Requirement 4: Feedback to the LLM on realized coverage

**User Story:** As a trainer, I want the next evolve prompt to show the
LLM both its declared MassBudget and the realized coverage from the
previous round, so that the LLM stops repeating the same intent-vs-
realization gap.

#### Acceptance Criteria

1. WHEN constructing the evolve prompt, THE PromptBuilder SHALL include a
   "Previous-round MassBudget vs realized coverage" section listing every
   HighLiftCell with the triplet `(target_probability,
   realized_probability, deviation)`.
2. WHEN the previous Generator was rejected by DryRunValidator, THE
   PromptBuilder SHALL include the rejection reason and the offending
   cells in the prompt body.
3. THE PromptBuilder SHALL preserve the existing `attribution`,
   `coverage`, `history`, and `cfa` evidence sections so that
   `feedback_mode=attribution` and `feedback_mode=cfa` continue to render
   as today.
4. WHERE `curriculum.controller.enabled=False`, THE PromptBuilder SHALL
   render the prompt byte-identically to the current implementation for
   the same inputs.

### Requirement 5: Structured-output proposer mode

**User Story:** As a trainer, I want to optionally constrain the LLM to a
JSON proposal schema instead of free-form Python, so that the controller
can compile proposals deterministically and side-step the procedural
fallback waterfall.

#### Acceptance Criteria

1. WHERE `curriculum.controller.proposer_mode=structured`, THE ACGS_API
   SHALL prompt the LLM for a JSON object whose schema includes per-cell
   target weights and per-cell geometric template parameters, and SHALL
   document the schema in the spec design document.
2. THE GeneratorProposalParser SHALL parse the LLM JSON output into a
   typed `GeneratorProposal` Python object and SHALL return a structured
   error when the JSON is invalid against the schema.
3. THE GeneratorProposalSerializer SHALL serialize a `GeneratorProposal`
   back into the same JSON form.
4. FOR ALL valid `GeneratorProposal` objects, parsing then serializing
   then parsing SHALL produce an equivalent `GeneratorProposal`
   (round-trip property).
5. IF the LLM JSON output fails schema validation, THEN THE ACGS_API
   SHALL re-prompt up to `ACGS_API.max_evolve_retries` with the
   validation error appended to the user prompt.
6. THE GeneratorCompiler SHALL compile a `GeneratorProposal` into a
   Generator whose realized empirical distribution matches the proposed
   per-cell weights within `controller.mass_tolerance` when measured by
   DryRunValidator.
7. WHERE `curriculum.controller.proposer_mode=freeform`, THE ACGS_API
   SHALL prompt the LLM for free-form Python code as the current
   implementation does.

### Requirement 6: Decoupled "what" vs "how" mode

**User Story:** As a trainer, I want the controller to fix the per-cell
mass distribution while the LLM only contributes per-cell geometric
primitives, so that the LLM cannot redirect mass between cells through
fallback chains.

#### Acceptance Criteria

1. WHERE `curriculum.controller.proposer_mode=cell_geometry`, THE
   CurriculumController SHALL fix the per-cell weights from the
   deterministic MassBudget and SHALL pass only the cell list and
   per-cell sampling envelopes to the LLM.
2. WHEN producing a layout, THE Sampler SHALL first draw a Cell from
   MassBudget, then SHALL invoke the LLM-supplied per-cell geometric
   primitive to draw `(x, y)` within that Cell.
3. IF the LLM-supplied per-cell primitive for a Cell cannot return a
   layout point that passes `is_safe`, in-bounds, and inter-obstacle
   separation within `controller.per_cell_max_attempts` attempts (default
   50), THEN THE Sampler SHALL fall back to a deterministic in-cell
   rejection sampler defined by the controller and SHALL log the
   substitution event to `train.log` at WARNING level with key
   `controller.cell_geometry_fallback`.
4. THE Sampler SHALL include in each obstacle's return record an
   `intended_cell_id` field so DryRunValidator can detect intent-vs-
   realized mismatches at the per-call level.

### Requirement 7: Per-cell / lift-weighted DifficultySignal

**User Story:** As a trainer, I want curriculum decisions to be driven by
the policy's failure rate on HighLiftCells rather than by global success
rate, so that "too_easy" is no longer triggered while the policy still
fails ~87% of the time on hard cells.

#### Acceptance Criteria

1. THE CurriculumController SHALL compute `policy_sr_high_lift` as the
   per-episode success rate restricted to episodes whose obstacle layout
   contains at least one obstacle whose encoded `cell_id` is in
   HighLiftCells.
2. THE DifficultySignal SHALL also include the lift-weighted aggregate
   `sum(failure_lift_c * failure_rate_c) / sum(failure_lift_c)` over
   Cells with `total_count >= AttributionConfig.min_support`.
3. WHEN `policy_sr_high_lift > controller.high_lift_sr_high` (default
   0.7), THE CurriculumController SHALL classify the round as
   `too_easy_on_hard_cells` and SHALL increase HighLiftCell minimum
   probabilities for the next MassBudget by a configurable step
   `controller.mass_step_up`.
4. WHEN `policy_sr_high_lift < controller.high_lift_sr_low` (default
   0.2), THE CurriculumController SHALL classify the round as
   `too_hard_on_hard_cells` and SHALL decrease HighLiftCell minimum
   probabilities for the next MassBudget by a configurable step
   `controller.mass_step_down`.
5. WHERE `curriculum.controller.signal_mode=global_sr`, THE
   CurriculumController SHALL fall back to the existing `_difficulty_reason`
   in `TD3CurriculumWorkspace` so prior experiments remain comparable.

### Requirement 8: Population / ensemble accumulation of generators

**User Story:** As a trainer, I want previously accepted Generators to be
retained in a mixture rather than replaced, so that hard regimes uncovered
in earlier evolutions are not forgotten in later rounds.

#### Acceptance Criteria

1. WHERE `curriculum.controller.population_mode=enabled`, THE
   CurriculumController SHALL maintain an ordered list of accepted
   Generators with per-Generator mixture weights.
2. WHEN a new Generator is accepted, THE CurriculumController SHALL set
   that Generator's mixture weight inversely proportional to its
   DryRun-measured `policy_sr_high_lift`.
3. WHEN `_make_llm_scene` requests a layout, THE Sampler SHALL first draw
   a Generator from the mixture according to current weights, then SHALL
   draw obstacles from that Generator.
4. THE CurriculumController SHALL persist per-round mixture weights to
   `<output_dir>/controller/evolve_NNN/population.json`.
5. WHERE `curriculum.controller.population_mode=disabled`, THE
   TD3CurriculumWorkspace SHALL behave as the current single-generator
   implementation: a new accepted Generator replaces the previous one.

### Requirement 9: Adversarial framing with FeasibilityOracle

**User Story:** As a trainer, I want the LLM's proposal objective to be
"maximize policy regret subject to FeasibilityOracle solvability" rather
than "make harder while feasible", so that the proposer is no longer
biased toward preserving overall solvability at the cost of HighLiftCell
coverage.

#### Acceptance Criteria

1. WHERE `curriculum.controller.framing=adversarial`, THE PromptBuilder
   SHALL render the proposer objective as "maximize per-cell policy
   failure rate on HighLiftCells while every layout passes the
   FeasibilityOracle".
2. THE FeasibilityOracle SHALL be a deterministic solvability check over
   the corridor between `tblock_pose` and target pose with the obstacles
   in place; the design document chooses the implementation strategy
   (BFS over a discretized configuration grid is the canonical option).
3. WHEN computing CoverageReport, THE DryRunValidator SHALL invoke the
   FeasibilityOracle on each generated layout and SHALL record the
   per-layout feasibility outcome.
4. IF the fraction of FeasibilityOracle-solvable layouts in DryRun is
   below `controller.min_feasibility_rate` (default 0.95), THEN THE
   DryRunValidator SHALL reject the Generator candidate.
5. WHERE `curriculum.controller.framing=cooperative`, THE PromptBuilder
   SHALL render the current "harder while feasible" objective so prior
   experiments remain reproducible.

### Requirement 10: Backward compatibility

**User Story:** As a trainer with an existing experimental harness, I
want the new control mechanism to be opt-in via the existing Hydra
config tree, so that current `td3_curriculum_workspace`, `attribution`,
and `acgs_api` modules keep running unmodified when the controller is
disabled.

#### Acceptance Criteria

1. WHERE `curriculum.controller.enabled=False`, THE TD3CurriculumWorkspace
   SHALL execute the current evolve loop with no behavioral change
   measured at the byte level on `train.log`, on prompt artifacts, and on
   accepted generator code for the same seed and the same configuration
   that the diagnostic run used.
2. THE public APIs `compute_attribution`, `AttributionConfig`, and
   `write_attribution_outputs` in `DIVO/curriculum/attribution.py` SHALL
   remain unchanged in signature and in output schema.
3. THE public methods `ACGS_API.evolve` and `ACGS_API.get_prompt_text`
   SHALL remain unchanged in signature; new behavior SHALL be added
   through new keyword arguments whose defaults preserve current
   behavior.
4. THE existing checkpoint payload (`CurriculumRuntimeState` plus
   `RLLoopState`) SHALL remain readable; new controller state SHALL be
   stored under a new optional key whose absence triggers default
   initialization.
5. WHEN loading a checkpoint that predates the controller, THE
   TD3CurriculumWorkspace SHALL resume training without raising and
   SHALL initialize controller state from the loaded
   `attribution_history` if available, otherwise from a uniform
   MassBudget.

### Requirement 11: Reproducibility

**User Story:** As a trainer, I want the controller, dry-run, and
sampler to be fully reproducible given a seed, so that experimental
comparisons remain deterministic across runs.

#### Acceptance Criteria

1. THE CurriculumController SHALL accept `controller.seed` from the
   Hydra config and SHALL derive every internal RNG stream from that
   seed.
2. WHEN given the same `controller.seed`, the same AttributionResult,
   and the same GeneratorProposal, THE DryRunValidator SHALL produce a
   byte-identical CoverageReport.
3. THE Sampler SHALL accept a per-call seed argument so that
   `_make_llm_scene` can replay layouts deterministically given the same
   input pose.
4. THE CurriculumController SHALL record every RNG seed used per
   EvolveRound in `<output_dir>/controller/evolve_NNN/seeds.json`,
   including the dry-run seed, the sampler seed, and the per-cell
   primitive seeds when `proposer_mode=cell_geometry`.

### Requirement 12: Logging and diagnostics

**User Story:** As a trainer investigating a curriculum failure mode, I
want every EvolveRound to record both the declared intent and the
realized empirical distribution side by side, so that the failure mode
in `21.58.13_td3_pusht_llm_curriculum` is detectable from logs alone
without re-running attribution by hand.

#### Acceptance Criteria

1. THE TD3CurriculumWorkspace SHALL write per-round artifacts under
   `<output_dir>/controller/evolve_NNN/` containing at minimum:
   `mass_budget.json`, `coverage_report.json`, `per_cell_deviation.csv`,
   `decision.json`, and `seeds.json`.
2. WHEN a Generator candidate is rejected by DryRunValidator, THE
   TD3CurriculumWorkspace SHALL persist the rejected proposal source
   (Python or JSON, as applicable) under
   `<output_dir>/controller/evolve_NNN/rejected/proposal_K.{py,json}`.
3. THE TD3CurriculumWorkspace SHALL emit at INFO level a one-line
   summary per round with the fields `evolve_index`, `accepted`,
   `policy_sr_high_lift`, `max_per_cell_deviation`, `signal_class`,
   and `n_high_lift_cells`.
4. WHERE `curriculum.controller.enabled=True`, THE wandb log SHALL emit
   the metrics `controller/policy_sr_high_lift`,
   `controller/max_per_cell_deviation`,
   `controller/n_rejected_proposals`, and
   `controller/n_high_lift_cells` once per EvolveRound.
