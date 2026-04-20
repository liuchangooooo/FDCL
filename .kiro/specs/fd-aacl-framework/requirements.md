# Requirements Document: Failure-Driven Adaptive Adversarial Curriculum Learning (FD-AACL) Framework

## Introduction

The FD-AACL framework is a sophisticated adversarial curriculum learning system for the Push-T robotic manipulation task. Instead of generating specific obstacle coordinates, the system generates reusable topology generators (Python code) that adapt to policy weaknesses discovered through autonomous failure analysis. The framework implements a complete adversarial game loop where the LLM analyzes failure patterns, generates adversarial environments targeting those weaknesses, and the policy improves through training on progressively harder scenarios.

## Glossary

- **System**: The FD-AACL framework including all 6 stages
- **Topology_Generator**: Python function that generates obstacle configurations given a T-block pose
- **Failure_Snapshot**: Compressed data capturing the last 5-10 steps before policy failure
- **Failure_Graph**: LLM-constructed representation of key geometric configurations that trigger failures
- **Strategy_Executor**: Safe execution environment for LLM-generated Python code
- **Policy**: The TD3 reinforcement learning agent being trained
- **T-block**: The asymmetric object being manipulated (horizontal bar 0.10m×0.03m, vertical bar 0.03m×0.07m)
- **Workspace**: 0.5m × 0.5m area with coordinates [-0.25, 0.25]
- **Obstacle**: 0.02m × 0.02m square blocks
- **Success_Rate**: Percentage of episodes where policy achieves goal (target: 60-75%)
- **Collision_Threshold**: 0.08m for training, 0.04m for evaluation
- **Target_Pose**: Fixed goal position at (0, 0, -45°)
- **Adversarial_Loop**: Iterative process where policy improvement triggers new failure analysis
- **Semantic_Attribution**: Converting numerical failure data to natural language descriptions
- **Hard_Mutation**: Preserving failure-inducing geometric configurations while adding luring elements
- **Solvability_Check**: Validation that generated environment has at least one feasible solution path

## Requirements

### Requirement 1: Stage 0 - Cold Start Bootstrapping

**User Story:** As a curriculum designer, I want to generate simple initial topology generators, so that the policy can learn basic skills and accumulate initial training data.

#### Acceptance Criteria

1. WHEN the system starts Stage 0, THE System SHALL generate a Python topology generator function with easy difficulty
2. THE Topology_Generator SHALL accept tblock_pose as input and return obstacle configurations
3. THE Topology_Generator SHALL place obstacles on path sides without blocking the direct path
4. THE Topology_Generator SHALL be parameterized to work across different tblock_pose values
5. THE System SHALL validate the generated topology generator through physical checks
6. IF validation fails, THEN THE System SHALL request LLM regeneration with feedback (max 3 retries)

### Requirement 2: Stage 1 - Training and Failure Snapshot Capture

**User Story:** As a system operator, I want to capture failure snapshots during training, so that I can analyze policy weaknesses without storing full episode data.

#### Acceptance Criteria

1. WHEN training 100 episodes with current topology generator, THE System SHALL capture death snapshots for failed episodes
2. THE Failure_Snapshot SHALL contain only the last 5-10 steps before failure
3. THE Failure_Snapshot SHALL record T-block state, collision info, obstacle config, and trajectory features
4. THE System SHALL compress failure data to less than 1KB per failure
5. THE System SHALL store 20-50 failure cases for analysis
6. THE System SHALL track success rate, collision rate, average reward, and average steps

### Requirement 3: Stage 2 - Semantic Attribution

**User Story:** As a failure analyst, I want to convert numerical failure data to natural language, so that the LLM can understand and analyze failure patterns.

#### Acceptance Criteria

1. WHEN processing failure snapshots, THE System SHALL convert numerical data to structured natural language reports
2. THE System SHALL describe scene setup (T-block initial pose, target pose, obstacle configuration)
3. THE System SHALL describe failure process (trajectory, collision point, final state)
4. THE System SHALL perform geometric analysis (distances, angles, spatial relationships)
5. THE System SHALL infer failure reason based on geometric features
6. THE System SHALL format reports to be token-efficient for LLM consumption

### Requirement 4: Stage 3 - LLM Autonomous Failure Analysis

**User Story:** As an AI researcher, I want the LLM to autonomously discover failure patterns, so that the system can identify policy weaknesses without predefined dimensions.

#### Acceptance Criteria

1. THE System SHALL NOT use predefined failure dimensions (e.g., "narrow corridor", "rotation")
2. WHEN analyzing failure reports, THE LLM SHALL autonomously identify 2-3 distinct failure modes
3. THE LLM SHALL name each failure mode and analyze its frequency
4. THE LLM SHALL identify geometric features associated with each failure mode
5. THE LLM SHALL construct a Failure_Graph with key geometric configs and trigger conditions
6. THE LLM SHALL perform root cause analysis distinguishing obstacle layout vs policy capability issues
7. THE LLM SHALL identify relationships between different failure modes

### Requirement 5: Stage 4 - Adversarial Topology Generator Creation

**User Story:** As a curriculum designer, I want the LLM to generate adversarial topology generators, so that the policy is trained on environments targeting its specific weaknesses.

#### Acceptance Criteria

1. WHEN generating adversarial topology, THE LLM SHALL extract key geometric configs from the Failure_Graph
2. THE LLM SHALL use hard-mutation generation preserving failure configs
3. THE LLM SHALL add luring elements to induce policy into failure configurations
4. THE LLM SHALL combine multiple failure modes in a single topology generator
5. THE LLM SHALL target "just at failure boundary" difficulty (success rate 60-75%)
6. IF success rate > 80%, THEN THE System SHALL increase difficulty
7. IF success rate < 50%, THEN THE System SHALL decrease difficulty
8. THE LLM SHALL output Python code (topology generator function), not JSON coordinates
9. THE Topology_Generator SHALL be reusable across multiple episodes with different tblock_pose values

### Requirement 6: Stage 5 - Execution and Validation

**User Story:** As a safety engineer, I want to validate LLM-generated code, so that only safe and solvable environments are used for training.

#### Acceptance Criteria

1. WHEN loading LLM-generated topology generator, THE Strategy_Executor SHALL execute code in a sandboxed environment
2. THE System SHALL perform physical validation checking for collisions
3. THE System SHALL perform solvability check ensuring at least one feasible path exists
4. IF validation fails, THEN THE System SHALL provide feedback to LLM for regeneration
5. THE System SHALL allow maximum 3 retry attempts for validation failures
6. THE System SHALL log validation results for debugging

### Requirement 7: Stage 6 - Adversarial Game Loop

**User Story:** As a curriculum manager, I want the system to automatically trigger new analysis cycles, so that the curriculum continuously adapts as the policy improves.

#### Acceptance Criteria

1. WHEN policy success rate increases above 80%, THE System SHALL trigger new failure analysis
2. WHEN new failure modes emerge, THE System SHALL analyze them and generate new adversarial environments
3. THE System SHALL loop back to Stage 1 after generating new topology generator
4. THE System SHALL maintain history of topology generators and their performance metrics
5. THE System SHALL track curriculum progression across multiple cycles
6. THE System SHALL detect when policy has overcome old failure modes

### Requirement 8: Failure Snapshot System Implementation

**User Story:** As a data engineer, I want an efficient failure snapshot system, so that I can capture meaningful data without excessive storage overhead.

#### Acceptance Criteria

1. THE System SHALL capture snapshots only when episode fails (reward < success threshold)
2. THE Failure_Snapshot SHALL include T-block pose (x, y, θ) for last 5-10 steps
3. THE Failure_Snapshot SHALL include collision flag and collision position if applicable
4. THE Failure_Snapshot SHALL include obstacle configuration (positions only, not full state)
5. THE Failure_Snapshot SHALL include trajectory features (path length, curvature, velocity)
6. THE System SHALL compress snapshot data to < 1KB per failure
7. THE System SHALL store snapshots in memory-efficient format (numpy arrays or compressed JSON)

### Requirement 9: Topology Generator Format and Constraints

**User Story:** As a code generator, I want clear format specifications for topology generators, so that generated code is consistent and safe.

#### Acceptance Criteria

1. THE Topology_Generator SHALL be a Python function with signature: `generate_obstacles(tblock_pose: List[float], num_obstacles: int) -> List[Dict]`
2. THE Topology_Generator SHALL accept tblock_pose as [x, y, θ] in meters and radians
3. THE Topology_Generator SHALL return list of dicts with keys: 'x', 'y', 'purpose'
4. THE Topology_Generator SHALL ensure all coordinates are within [-0.2, 0.2] range
5. THE Topology_Generator SHALL ensure obstacles are > 0.12m from T-block centers (start and target)
6. THE Topology_Generator SHALL ensure obstacles are > 0.03m apart from each other
7. THE Topology_Generator SHALL NOT use external libraries beyond numpy and math
8. THE Topology_Generator SHALL be deterministic given the same tblock_pose and random seed

### Requirement 10: Strategy Executor Safety

**User Story:** As a security engineer, I want safe execution of LLM-generated code, so that malicious or buggy code cannot harm the system.

#### Acceptance Criteria

1. THE Strategy_Executor SHALL execute topology generators in a restricted Python environment
2. THE Strategy_Executor SHALL whitelist only safe modules (numpy, math, random)
3. THE Strategy_Executor SHALL enforce execution timeout (5 seconds maximum)
4. THE Strategy_Executor SHALL catch and log all exceptions during execution
5. THE Strategy_Executor SHALL validate output format before returning results
6. IF execution fails or times out, THEN THE Strategy_Executor SHALL return error details for LLM feedback

### Requirement 11: LLM Prompt Design for Failure Analysis

**User Story:** As a prompt engineer, I want effective prompts for autonomous failure analysis, so that the LLM discovers meaningful patterns without predefined dimensions.

#### Acceptance Criteria

1. THE System SHALL provide failure reports in structured natural language format
2. THE System SHALL NOT mention predefined failure categories in prompts
3. THE System SHALL ask LLM to identify patterns autonomously from raw failure data
4. THE System SHALL request LLM to name discovered failure modes
5. THE System SHALL request LLM to construct Failure_Graph with geometric relationships
6. THE System SHALL request LLM to perform root cause analysis
7. THE System SHALL provide examples of good failure analysis (few-shot learning)

### Requirement 12: LLM Prompt Design for Topology Generation

**User Story:** As a prompt engineer, I want effective prompts for topology generator creation, so that the LLM generates valid, adversarial Python code.

#### Acceptance Criteria

1. THE System SHALL provide Failure_Graph and key geometric configs to LLM
2. THE System SHALL specify Python function signature and constraints
3. THE System SHALL provide examples of valid topology generator code
4. THE System SHALL specify safety constraints (distance thresholds, coordinate ranges)
5. THE System SHALL request code that targets specific failure modes
6. THE System SHALL request code that is reusable across different tblock_pose values
7. THE System SHALL specify target success rate (60-75%)

### Requirement 13: Integration with Existing Training System

**User Story:** As a system integrator, I want seamless integration with the existing TD3 training workspace, so that the FD-AACL framework enhances rather than replaces current training.

#### Acceptance Criteria

1. THE System SHALL integrate with TD3CurriculumWorkspace class
2. THE System SHALL use existing environment interface (set_obstacle_config method)
3. THE System SHALL collect training statistics (success rate, collision rate, rewards)
4. THE System SHALL trigger topology generator updates based on training progress
5. THE System SHALL maintain backward compatibility with existing LLM obstacle generator
6. THE System SHALL log curriculum progression to wandb for monitoring

### Requirement 14: Validation System Implementation

**User Story:** As a quality assurance engineer, I want comprehensive validation of generated environments, so that only high-quality, solvable configurations are used.

#### Acceptance Criteria

1. WHEN validating topology generator output, THE System SHALL check collision constraints
2. THE System SHALL verify obstacles are > 0.12m from start T-block center
3. THE System SHALL verify obstacles are > 0.12m from target T-block center
4. THE System SHALL verify obstacles are > 0.03m apart from each other
5. THE System SHALL verify at least one path exists from start to target with width ≥ 0.11m
6. THE System SHALL use A* or similar pathfinding to verify solvability
7. IF validation fails, THEN THE System SHALL generate detailed feedback explaining which constraint was violated

### Requirement 15: Curriculum Progression Tracking

**User Story:** As a researcher, I want to track curriculum progression over time, so that I can analyze the effectiveness of the FD-AACL framework.

#### Acceptance Criteria

1. THE System SHALL log each topology generator with timestamp and stage number
2. THE System SHALL log policy performance metrics for each topology generator
3. THE System SHALL log identified failure modes and their frequencies
4. THE System SHALL log validation results (pass/fail, retry count)
5. THE System SHALL compute and log curriculum difficulty metrics over time
6. THE System SHALL save topology generator code to disk for reproducibility
7. THE System SHALL generate summary reports showing curriculum evolution

### Requirement 16: Adaptive Difficulty Control

**User Story:** As a curriculum designer, I want automatic difficulty adjustment, so that the policy is always challenged at the appropriate level.

#### Acceptance Criteria

1. WHEN success rate > 80% for 50 consecutive episodes, THE System SHALL increase difficulty
2. WHEN success rate < 50% for 50 consecutive episodes, THE System SHALL decrease difficulty
3. THE System SHALL adjust difficulty by modifying geometric constraints in topology generator
4. THE System SHALL maintain target success rate between 60-75%
5. THE System SHALL smooth difficulty adjustments to avoid sudden jumps
6. THE System SHALL log difficulty adjustment decisions and rationale

### Requirement 17: Failure Mode Diversity

**User Story:** As a curriculum designer, I want diverse failure modes to be discovered, so that the policy develops robust capabilities across different scenarios.

#### Acceptance Criteria

1. THE System SHALL track diversity of discovered failure modes over time
2. THE System SHALL encourage LLM to identify novel failure patterns
3. THE System SHALL avoid generating redundant topology generators targeting the same failure mode
4. THE System SHALL combine multiple failure modes in advanced stages
5. THE System SHALL maintain a library of discovered failure modes for reference

### Requirement 18: Serialization and Persistence

**User Story:** As a system administrator, I want to save and load curriculum state, so that training can be resumed after interruption.

#### Acceptance Criteria

1. THE System SHALL serialize topology generators to disk as Python files
2. THE System SHALL serialize failure snapshots to compressed format
3. THE System SHALL serialize Failure_Graph and analysis results to JSON
4. THE System SHALL save curriculum progression history
5. THE System SHALL support loading saved curriculum state on restart
6. THE System SHALL validate loaded topology generators before use

### Requirement 19: Monitoring and Debugging

**User Story:** As a developer, I want comprehensive logging and debugging tools, so that I can diagnose issues in the FD-AACL framework.

#### Acceptance Criteria

1. THE System SHALL log all LLM API calls with prompts and responses
2. THE System SHALL log topology generator execution results
3. THE System SHALL log validation failures with detailed error messages
4. THE System SHALL provide visualization of failure snapshots
5. THE System SHALL provide visualization of generated obstacle configurations
6. THE System SHALL log timing information for each stage
7. THE System SHALL support debug mode with verbose output

### Requirement 20: Performance Optimization

**User Story:** As a performance engineer, I want efficient implementation of the FD-AACL framework, so that it does not significantly slow down training.

#### Acceptance Criteria

1. THE System SHALL cache topology generator outputs for reuse across episodes
2. THE System SHALL batch failure snapshot processing
3. THE System SHALL limit LLM API calls to necessary stages only
4. THE System SHALL use efficient data structures for failure storage
5. THE System SHALL parallelize validation checks when possible
6. THE System SHALL minimize overhead in training loop (< 5% slowdown)
