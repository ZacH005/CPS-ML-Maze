# Final Demo Staged Roadmap

## Purpose

This document defines the stage-by-stage workflow for finishing the autonomous maze demo, paper, and presentation without shortcuts.

Use this document in a fresh AI chat by pasting it and saying:

```text
We are working on Stage <N>: <stage name>.
Do not implement yet. First inspect the relevant files, ask for any required human inputs, then create a concrete plan for this stage only. After I approve the plan, execute it, require any manual validation listed in the stage, update an agent log, and stop.
```

The goal is not just to make the maze work once. The goal is to produce a reliable, explainable cyber-physical system with evidence strong enough for a final paper and presentation.

## Global Rules

- Functionality before speed.
- Tracking reliability before controller tuning.
- Calibration consistency before path or hole tuning.
- Axis mapping before autonomous servo runs.
- One variable per test run.
- No silent guesses for hardware measurements, clicked points, ROI polygons, camera settings, or observed behavior.
- No workarounds for required human validation.
- No widening servo limits to compensate for tracking, path, or controller mistakes.
- Every meaningful change gets an agent log in `logs/agent/`.
- Stop at the end of the current stage. Do not drift into the next stage.

## Required Workflow For Every Stage

Each stage must follow this exact workflow:

1. Read the stage description and relevant files.
2. Inspect current repo state with `git status --short`.
3. Ask only the human-input questions required for that stage.
4. After answers, create a concrete implementation and validation plan for that stage only.
5. Explicitly check the plan against the stage acceptance criteria.
6. Execute only after the plan is approved.
7. Run non-hardware tests where possible.
8. Require manual/hardware validation when the stage lists it.
9. Update or create an agent log in `logs/agent/`.
10. Report:
    - what changed
    - what validation passed
    - what still needs manual validation
    - whether the stage acceptance criteria are satisfied

## Current Technical Strategy

Primary path:

- Fixed camera.
- Homography into inside playable maze coordinates, `263 mm x 222 mm`.
- Pipeline tracker using motion, specular highlights, ROI, and static confusers.
- Click-to-seed for demo reliability.
- Axis-map calibration for servo sign/swap behavior.
- Conservative path follower first.
- Speed tiers only after repeatability.

Single-entry recalibration:

```bash
python3 scripts/recalibrate_all.py
```

This runs the fixed-camera calibration sequence in order and opens
`scripts/launcher.py` during the recording step so the board can be manually
controlled while tracker/confuser footage is captured. Use `--video
data/raw/<recording>.avi` to reuse an existing recording instead of making a
new one.

Fallback path:

- If homography or full path becomes unreliable close to deadline, demonstrate a reliable short section with the same perception, control, safety, and logging stack.

## Stage 0 - Context And Artifact Audit

### Goal

Get the chat and repo synchronized before changing anything.

### Relevant Files

- `AGENTS.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/ARCHITECTURE.md`
- `docs/HARDWARE_NOTES.md`
- `docs/PATH_TO_SUCCESS.md`
- `docs/TUNING_ROADMAP.md`
- `docs/FINAL_DEMO_STAGED_ROADMAP.md`
- `configs/default.yaml`
- `configs/local.yaml`
- `calibration/README.md`

### Human Inputs Required

- Confirm which machine is being used.
- Confirm camera and serial devices in `configs/local.yaml`.
- Confirm the current blocker in plain language.

### Development Work

- No code changes by default.
- Summarize current artifacts and whether they exist:
  - `calibration/board_homography.npz`
  - `calibration/live_roi.json`
  - `calibration/live_confusers.json`
  - `configs/maze_holes.csv`
  - `configs/maze_path_auto.csv`
  - `calibration/axis_map.npz`
  - latest run log

### Acceptance Criteria

- The active machine, camera, serial port, coordinate frame, and current blocker are known.
- The next stage is chosen intentionally.

## Stage 1 - Calibration And Coordinate Freeze

### Goal

Ensure all derived artifacts use the same camera pose, flip convention, homography, and inside playable coordinate frame.

### Relevant Files

- `configs/default.yaml`
- `configs/local.yaml`
- `src/cps_maze/camera.py`
- `scripts/check_camera.py`
- `scripts/calibrate_homography.py`
- `scripts/auto_detect_holes.py`
- `scripts/auto_trace_path.py`
- `calibration/board_homography.npz`
- `configs/maze_holes.csv`
- `configs/maze_path_auto.csv`

### Human Inputs Required

- Confirm live camera view still matches the fixed reference orientation.
- Confirm inside playable dimensions are still `263 mm x 222 mm`.
- Click inside playable corners if homography must be regenerated.
- Approve the homography grid overlay.
- Approve detected holes and traced path.

### Development Work

- Reconfirm camera convention.
- Regenerate homography if needed.
- Regenerate holes and path if the homography or maze dimensions changed.
- Verify path/hole coordinate ranges fit the inside playable frame.

### Acceptance Criteria

- Homography grid hugs the inside playable maze area.
- Path and holes use the same coordinate frame.
- Path and hole coordinates are approximately within `0..263` and `0..222`.
- No autonomous tuning starts until this passes.

## Stage 2 - Tracking Reliability

### Goal

Make ball tracking stable enough that controller tuning is meaningful.

### Relevant Files

- `scripts/select_maze_roi.py`
- `scripts/pipeline.py`
- `src/cps_maze/vision/ball_pipeline.py`
- `configs/default.yaml`
- `calibration/live_roi.json`
- `calibration/live_confusers.json`
- Stage 1 recording in `data/raw/`

### Human Inputs Required

- Provide or approve the recording used for confuser calibration.
- Click or approve the ROI polygon.
- Confirm ROI excludes external ChArUco board, outer frame, desk, and background.
- Confirm annotated tracking video does not persistently lock onto holes, text, glare, or the ChArUco board.
- Provide observed ball/glare brightness if `vision.min_specular` needs tuning.

### Development Work

- Generate or update ROI.
- Generate or update static confusers.
- Compare tracking with and without confusers.
- Tune `vision.min_specular` only after visual evidence.
- Keep click-to-seed as the demo default unless auto-seed is proven reliable.

### Acceptance Criteria

- `calibration/live_roi.json` exists.
- `calibration/live_confusers.json` exists.
- ROI excludes non-playable regions.
- Tracker follows the ball through representative video/live motion.
- Lost-ball events lead to neutral behavior, not blind driving.

## Stage 3 - Servo Safety And Axis Mapping

### Goal

Prove that board-frame control commands map to the correct servo directions safely.

### Relevant Files

- `scripts/axis_check.py`
- `src/cps_maze/control/axis_map.py`
- `src/cps_maze/hardware/serial_link.py`
- `firmware/arduino/maze_servo_controller/maze_servo_controller.ino`
- `calibration/axis_map.npz`
- `configs/default.yaml`
- `configs/local.yaml`

### Human Inputs Required

- Confirm serial port.
- Place ball in an open area away from holes/walls.
- Click ball to seed tracker for each pulse.
- Confirm small pulses are physically safe.
- Stop physically if motion is unsafe.

### Development Work

- Run `axis_check.py`.
- Save `calibration/axis_map.npz`.
- Confirm response matrix has clear dominant axes.
- Do not continue if the matrix is ambiguous.

### Acceptance Criteria

- `calibration/axis_map.npz` exists.
- Positive/negative yaw and pitch effects are known.
- Neutral behavior is verified after pulses.
- Dry-run/low-power command direction looks correct.

## Stage 4 - Conservative Section Solve

### Goal

Make one section boringly reliable before attempting full-maze speed.

### Relevant Files

- `scripts/run_autonomous.py`
- `scripts/analyze_run.py`
- `configs/maze_path_auto.csv`
- `configs/maze_holes.csv`
- `calibration/board_homography.npz`
- `calibration/axis_map.npz`
- `calibration/live_confusers.json`
- `src/cps_maze/control/pid.py`

### Human Inputs Required

- Place the ball at the section start.
- Click the ball to seed.
- Press Space to arm.
- Supervise physically.
- Report observed failure mode after each failed run.

### Development Work

- Start with dry-run overlay.
- Run with conservative command cap.
- Use logs to classify failures.
- Tune only one parameter per run.
- Prefer path fixes for corner/wall pinning, not gain changes.

### Acceptance Criteria

- One section completes at least 4/5 times.
- Detection rate is high enough to trust logs.
- Cross-track error is reasonable for the physical channel.
- Failures are explainable from logs/video.

## Stage 5 - Full Maze Conservative Solve

### Goal

Complete the full route at conservative speed with safety and repeatability.

### Relevant Files

- Same as Stage 4
- best run logs in `data/raw/`
- processed videos/logs in `data/processed/`

### Human Inputs Required

- Confirm full-route path overlay is physically plausible.
- Place ball at start.
- Supervise each run.
- Classify failures: tracking, axis, path geometry, hole, wall, mechanical, speed, or unknown.

### Development Work

- Run full maze at conservative speed.
- Use `analyze_run.py` after every run.
- Fix recurring geometry failures in path/hole annotations.
- Keep speed low until success is repeatable.

### Acceptance Criteria

- Full maze succeeds at least once conservatively, or the best section-level fallback is stable and documented.
- At least 3 attempts are logged.
- Failure modes are classified.
- Demo command is known.

## Stage 6 - Tuning Ladder And Speed Tiers

### Goal

Increase speed only after conservative reliability is proven.

### Relevant Files

- `docs/TUNING_ROADMAP.md`
- `configs/default.yaml`
- `scripts/run_autonomous.py`
- `scripts/analyze_run.py`
- run logs and processed summaries

### Human Inputs Required

- Confirm conservative run is repeatable.
- Approve each speed tier before running it.
- Provide physical observations for overshoot, wall hits, holes, and instability.

### Development Work

- Create or document conservative, normal, and fast command presets.
- Run repeated trials per tier.
- Change one tuning variable at a time.
- Record success rate and completion time per tier.

### Acceptance Criteria

- There is a reliable conservative demo setting.
- Faster tiers are attempted only when lower tiers are stable.
- Evaluation table includes success rate, completion time, detection rate, and cross-track error.

## Stage 7 - Evidence Package For Paper And Presentation

### Goal

Turn the engineering work into defensible deliverables.

### Relevant Files

- `docs/PATH_TO_SUCCESS.md`
- `docs/ARCHITECTURE.md`
- `docs/HARDWARE_NOTES.md`
- `scripts/analyze_run.py`
- selected logs/videos/images

### Human Inputs Required

- Confirm final demo run(s) to use.
- Confirm paper/presentation requirements.
- Provide any rubric constraints.

### Development Work

- Select best logs and videos.
- Generate summary tables.
- Capture screenshots of ROI, path overlay, tracking overlay, and run metrics.
- Draft paper sections:
  - system architecture
  - calibration
  - perception
  - control
  - safety
  - evaluation
  - limitations

### Acceptance Criteria

- The project has a reproducible demo command.
- The paper has quantitative evaluation.
- The presentation has a live-demo fallback plan.
- The limitations are honest and technically grounded.

## Stage 8 - Demo Freeze

### Goal

Protect the final working state.

### Relevant Files

- final config files
- final calibration artifacts
- final logs/videos
- paper/presentation materials

### Human Inputs Required

- Confirm the final demo mode.
- Confirm whether any last-minute change is worth the risk.

### Development Work

- No feature work unless the demo is broken.
- Save final configs and artifacts.
- Record final known-good command sequence.
- Prepare fallback demo command.

### Acceptance Criteria

- Demo can be run from a short checklist.
- Fallback demo is ready.
- No untested change is introduced after freeze.

## Failure Routing Table

When a problem repeats, route it back to the correct stage instead of tuning randomly.

| Repeating symptom | Return to stage | Why |
|---|---:|---|
| Camera orientation, resolution, or device changed | Stage 1 | All pixel/homography artifacts may be stale |
| Path/hole coordinates exceed playable frame | Stage 1 | Homography/path/holes are inconsistent |
| Path overlay shifted from maze walls | Stage 1 | Homography or path is stale |
| Tracker locks onto CharUco, holes, numbers, or glare | Stage 2 | Perception artifact problem |
| Tracker loses ball often while board moves | Stage 2 | Tracking/lighting/min-specular problem |
| Ball moves opposite the target direction | Stage 3 | Axis map/sign problem |
| Ball moves diagonally for a single-axis command | Stage 3 | Axis map/mechanical coupling problem |
| Ball follows straight section but fails corners | Stage 4 | Path geometry/lookahead problem |
| Ball falls into same hole repeatedly | Stage 4 or 5 | Path needs local bias away from hole |
| Oscillation around path with stable tracking | Stage 6 | Controller damping/tuning problem |
| Sluggish movement with stable tracking | Stage 6 | Gain/stiction/speed tuning problem |
| Faster run becomes unreliable | Stage 6 | Speed tier exceeded repeatable control envelope |
| Logs do not explain failure | Stage 7 | Instrumentation/evidence problem |

## Tuning Discipline

Only tune gains after these are true:

- Camera convention is fixed.
- Homography/path/holes are consistent.
- ROI/confusers are active.
- Tracker is reliable enough for logs to be meaningful.
- Axis map is trusted.
- The failure repeats in the same way.

Recommended tuning order:

1. Path geometry and lookahead for wall/corner failures.
2. `kd` for oscillation.
3. `kp` for sluggish target following.
4. `stall_kick` for stiction.
5. `ki` for small persistent bias.
6. `max_command` only for speed after reliability.

## New Chat Prompt Template

Use this prompt at the start of a new chat:

```text
You are working in /Users/zach/CPS-ML-Maze.

Read:
1. AGENTS.md
2. docs/PROJECT_CONTEXT.md
3. docs/ARCHITECTURE.md
4. docs/HARDWARE_NOTES.md
5. docs/PATH_TO_SUCCESS.md
6. docs/FINAL_DEMO_STAGED_ROADMAP.md

We are working on Stage <N>: <stage name>.

Do not implement yet. First inspect the relevant files listed for this stage and summarize the current state. Then ask only the human-input questions required for this stage. After I answer, create a concrete implementation and validation plan for this stage only. Explicitly check that the plan satisfies the stage acceptance criteria. Do not move to the next stage.
```

After approving the plan, use:

```text
Execute the approved Stage <N> plan.
Keep edits scoped to this stage.
Run tests, compile checks, or non-hardware validation where possible.
Require the listed manual/hardware validation; do not work around it.
Add or update an agent log in logs/agent/.
Stop after this stage and report whether the acceptance criteria passed.
```
