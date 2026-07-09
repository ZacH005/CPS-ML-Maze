# Live MVP Development Roadmap

## Purpose

This document is meant to be handed to a fresh AI coding chat. It defines the macro sequence for getting the physical maze to run a camera-feedback MVP.

The target MVP is:

> Detect the ball from the fixed camera, follow a short route through the maze, and send safe servo commands until the ball reaches the finish area or the run aborts safely.

Do not start with full-maze optimization. Do not block on ChArUco homography. Get one reliable closed-loop segment working, then expand.

## Required Context To Read First

Before making a plan or editing code, the next chat must read:

1. `AGENTS.md`
2. `docs/PROJECT_CONTEXT.md`
3. `docs/ARCHITECTURE.md`
4. `docs/HARDWARE_NOTES.md`
5. `docs/MANDATORY_MVP_NEXT_STEPS.md`
6. `docs/LIVE_MVP_DEVELOPMENT_ROADMAP.md`
7. The specific scripts/modules named in the current stage

Also inspect the current fixed camera view:

- `calibration/CURRENT_FIXED_CAMERA_VIEW.png`

This image shows the actual fixed camera perspective. It includes the maze, the side-mounted ChArUco board, and the current lighting/geometry constraints.

## Current Hardware And Setup Facts

- Environment: macOS.
- Camera position: mechanically fixed.
- Current fixed-camera reference image: `calibration/CURRENT_FIXED_CAMERA_VIEW.png`.
- Camera view orientation is accepted as the working orientation.
- In the camera view, the finish is near the bottom-left and the start is near the top-right.
- Board-frame origin for physical measurements is the bottom-left of the board.
- Firmware is functional.
- Servos have been actively tested through `scripts/launcher.py`.
- Manual keyboard/touchpad control works enough to move the ball.
- Servo axis mapping is not fully trusted.
- Existing `scripts/axis_check.py` may not be reliable yet because its current ball detection path has not tracked the ball well.
- A clicked homography has been recalibrated from the current fixed view, but there is concern that lens distortion makes straight projected lines not match visibly curved maze walls.
- The visible ChArUco board is a 5x5 ChArUco board using 6x6 ArUco tags (`DICT_6X6_100`).
- The ChArUco board is glued onto the side/outside of the board.
- The ChArUco board should not be treated as the primary ball-plane calibration target unless a later stage proves it is geometrically valid for that use.

## Brutal Calibration Position

The side-mounted ChArUco board is visible and detectable, but it is not the main path to the MVP.

Reason:

- A homography maps one plane.
- The ball rolls on the maze surface.
- The ChArUco board is glued onto the side/outside of the board.
- If the ChArUco plane is not the same as the rolling surface plane, it cannot directly define the ball-plane transform.

Use ChArUco only for:

- detector debugging
- optional intrinsics exploration
- future calibration experiments

Do not use it as a required dependency for the live control MVP.

## Development Workflow For Each Stage

For each stage below, use this workflow:

1. Ask for any human measurement, hardware action, or recording listed in "Human Inputs Required".
2. Create a concrete implementation plan for only that stage.
3. Review whether the plan satisfies the stage acceptance criteria.
4. Execute the plan.
5. Run non-hardware tests where possible.
6. Run the specified hardware/manual validation with the human.
7. Update or create an agent log in `logs/agent/`.
8. Do not proceed to the next stage until the current stage passes or is explicitly abandoned.

Do not silently invent measurements. If a stage needs a clicked point, video, ROI polygon, device index, camera mode, or manually observed behavior, ask for it.

## Stage 0 - Establish Fixed Camera And Runtime Conventions

### Goal

Make the camera convention explicit and prevent later scripts from mixing orientations, resolutions, or coordinate frames.

### Relevant Files

- `configs/default.yaml`
- `src/cps_maze/camera.py`
- `scripts/check_camera.py`
- `calibration/CURRENT_FIXED_CAMERA_VIEW.png`

### Human Inputs Required

- Confirm the live camera feed still matches `calibration/CURRENT_FIXED_CAMERA_VIEW.png`.
- Confirm the camera device index on macOS.
- Confirm whether `configs/default.yaml` currently displays the accepted orientation without additional flips.
- Confirm target camera mode: width, height, FPS.

### Development Work

- Verify `CameraCapture` uses the same camera settings and flip convention everywhere.
- Add documentation or operator checks if needed.
- Do not recalibrate or collect waypoints until this is stable.

### Acceptance Criteria

- A live camera check shows the same orientation as the fixed reference image.
- The coordinate convention is written down in the stage result.
- Any later capture/recording script will use the same camera settings.

## Stage 1 - Add A Native Live Camera Recording Script

### Goal

Record fixed-camera video directly from the camera, not from a screen recording, so pixel coordinates match live runtime.

This is required because `scripts/pipeline.py` can compute static confusers from a complete video, and the current best ball tracker came from `pipeline.py` testing on video.

### Relevant Files

- `src/cps_maze/camera.py`
- `configs/default.yaml`
- `scripts/pipeline.py`
- `data/raw/` or another agreed output directory

### Human Inputs Required

- Confirm where recordings should be saved.
- Confirm desired recording duration, defaulting to 60 seconds.
- During recording, the human should use `scripts/launcher.py` or existing teleop to move the ball around the playable maze area.

### Development Work

- Create a script that records from `CameraCapture(config.camera)` to a video file.
- Save metadata next to the video: config path, resolution, FPS requested, FPS observed if measured, timestamp, and flip settings.
- The script should not resize frames.
- The script should not screen-record.
- Include a preview window if useful, but do not let preview scaling affect saved frame pixels.

### Acceptance Criteria

- A 60 second video is saved from the fixed camera.
- The saved video frame size matches the runtime camera frame size.
- A still frame from the video visually matches `calibration/CURRENT_FIXED_CAMERA_VIEW.png`.
- The human confirms the ball moved through representative regions during recording.

## Stage 2 - Generate Static Confusers And Maze ROI For The Pipeline Tracker

### Goal

Create a static-confuser file and ROI that let the pipeline-style tracker ignore holes, glare, the side ChArUco board, desk/background, and other non-ball bright features.

### Relevant Files

- `scripts/pipeline.py`
- The video recorded in Stage 1
- Output confuser file, for example `calibration/live_confusers.json`
- `calibration/CURRENT_FIXED_CAMERA_VIEW.png`

### Human Inputs Required

- Provide or approve an ROI polygon around only the playable maze area.
- Confirm whether the side ChArUco board and outside frame are excluded by the ROI.
- Provide the Stage 1 recording path.

### Development Work

- Use the recorded video to generate static confusers.
- If existing `pipeline.py --calibrate` is insufficient for ROI entry, create a small helper workflow to click the ROI polygon on a frame.
- Ensure the output file is tied to the current camera pose and resolution.

### Acceptance Criteria

- `calibration/live_confusers.json` or equivalent exists.
- ROI excludes the side ChArUco board.
- Pipeline tracker run on the recorded video has fewer false locks on holes/glare than without confusers.

## Stage 3 - Extract Or Wrap The Pipeline Ball Tracker For Live Use

### Goal

Use the successful `scripts/pipeline.py` ball-tracking logic on live camera frames.

### Relevant Files

- `scripts/pipeline.py`
- `src/cps_maze/vision/ball_tracker.py`
- New module or script to be planned in this stage
- `src/cps_maze/camera.py`
- `configs/default.yaml`

### Human Inputs Required

- Provide initial seed location, either by clicking the live view or by placing the ball at a known visible start point.
- Confirm whether a click-to-seed UI is acceptable.
- Provide the confuser file from Stage 2.

### Development Work

- Prefer extracting the reusable `BallTracker` logic from `scripts/pipeline.py` into a module under `src/cps_maze/vision/`.
- Add a live preview script that reads from `CameraCapture`, applies ROI/confuser filtering, and displays ball state.
- The live tracker must report structured state: timestamp, `x_px`, `y_px`, radius, and status (`seed`, `detected`, `predicted`, `lost`).
- Do not couple this first live tracker to servo control.

### Acceptance Criteria

- The live tracker follows the ball in the current camera feed.
- It ignores the side ChArUco board.
- It does not persistently lock to holes, printed numbers, screws, wall highlights, or glare.
- When the ball is hidden or lost, status becomes `lost` or at least not confidently `detected`.
- A short CSV log can be written for analysis.

## Stage 4 - Decide Pixel-Space Or Homography-Space For First Control

### Goal

Choose the coordinate frame for the first closed-loop controller.

The default recommendation is pixel-space for the first working MVP unless the clicked homography is proven accurate enough for control.

### Relevant Files

- `scripts/calibrate_homography.py`
- `calibration/board_homography.npz`
- `scripts/annotate_path.py`
- `scripts/run_autonomous.py`
- Live tracker from Stage 3

### Human Inputs Required

- Confirm whether the current clicked homography overlay is good enough around the short first route segment.
- If homography is used, human must visually inspect the grid/path overlay from the current camera view.
- If pixel-space is used, human must provide or click pixel waypoints directly on the live frame.

### Development Work

Option A, recommended first:

- Use pixel coordinates for short-segment path following.
- Collect waypoints as `x_px,y_px`.
- Avoid lens distortion issues for the first MVP by staying in the same image frame used for tracking.

Option B, acceptable if validated:

- Use the current clicked homography.
- Limit the first test route to a region where the overlay is visually acceptable.
- Do not use the side ChArUco homography.

### Acceptance Criteria

- A short 3 to 5 waypoint route can be overlaid on the live camera frame.
- The route lies in the channel centerline.
- The chosen coordinate frame is explicitly documented.

## Stage 5 - Build A Short-Segment Waypoint Tool

### Goal

Create the minimum waypoint workflow for a short first segment, not the full maze.

### Relevant Files

- `scripts/annotate_path.py`
- `configs/`
- Live tracker from Stage 3
- Potential new pixel waypoint script

### Human Inputs Required

- Human clicks 3 to 5 waypoints for the first segment.
- Human confirms the route is physically reachable and not too close to holes/walls.

### Development Work

- If using pixel-space, create a pixel waypoint collection and overlay script.
- If using homography-space, adapt or use `scripts/annotate_path.py`, but only for a short segment.
- The tool must save the route in a format that the live controller can load.

### Acceptance Criteria

- Saved route overlays correctly on live camera frames.
- Route can be reloaded without shifting.
- Human signs off that the first route is feasible.

## Stage 6 - Replace Or Repair Axis Mapping For Live Tracker

### Goal

Determine how servo commands move the ball in the chosen coordinate frame.

The existing `scripts/axis_check.py` may be unreliable because it depends on ball detection that was not tracking well. This stage must use the Stage 3 live tracker or a direct human-observed fallback.

### Relevant Files

- `scripts/axis_check.py`
- `src/cps_maze/control/axis_map.py`
- `scripts/launcher.py`
- Firmware: `firmware/arduino/maze_servo_controller/maze_servo_controller.ino`

### Human Inputs Required

- Place the ball in an open area away from walls/holes.
- Confirm whether small pulses visibly move the ball.
- If automated tracking fails, human must report observed direction for +yaw, -yaw, +pitch, -pitch.

### Development Work

- Update or create an axis test that uses the live tracker from Stage 3.
- Test tiny safe servo pulses.
- Record direction mapping in config or calibration output.
- Do not widen servo limits to compensate for weak movement until mechanical safety is confirmed.

### Acceptance Criteria

- Positive/negative yaw and pitch effects are known in the selected coordinate frame.
- The controller can convert "move ball toward target" into servo yaw/pitch commands.
- Neutral behavior is verified after each test.

## Stage 7 - Build The First Live Closed-Loop Segment Runner

### Goal

Run a camera-feedback loop over a short 3 to 5 waypoint route.

### Relevant Files

- Live tracker from Stage 3
- Waypoint file from Stage 5
- Axis mapping from Stage 6
- `src/cps_maze/hardware/serial_link.py`
- `firmware/arduino/maze_servo_controller/maze_servo_controller.ino`
- Existing `scripts/run_autonomous.py` for reference

### Human Inputs Required

- Confirm serial port.
- Place the ball at the first waypoint/start region.
- Be ready to stop the run physically if needed.

### Development Work

- Implement a conservative live runner.
- Start with dry-run preview only.
- Then run with a very low command cap.
- Use simple proportional control first.
- Command neutral when status is not confidently detected.
- Command neutral on finish, timeout, camera failure, serial failure, or user abort.

### Acceptance Criteria

- Dry-run preview shows ball, target, error vector, and proposed command.
- Low-power live run moves the ball toward the first target.
- A 3 to 5 waypoint segment succeeds or fails with clear logs explaining why.
- No runaway servo behavior occurs.

## Stage 8 - Add Run Logging And Failure Replay

### Goal

Make failures diagnosable.

### Required Log Fields

- timestamp
- frame index if available
- tracker status
- ball `x_px,y_px`
- ball radius
- target `x,y` in chosen coordinate frame
- waypoint index
- raw error vector
- command before clamp
- command after clamp
- yaw command
- pitch command
- run state

### Human Inputs Required

- Provide a failed run log if behavior is bad.
- Describe the physical failure in plain language.

### Development Work

- Ensure every live run writes a CSV.
- Optionally save annotated debug video at low FPS or on failure.
- Add enough status strings to distinguish lost ball, near hole, target reached, timeout, and manual abort.

### Acceptance Criteria

- A failed run can be diagnosed from log plus the human's observation.
- The next adjustment can be made without guessing.

## Stage 9 - Expand From One Segment To A Maze Section

### Goal

Only after the short segment works, extend to a larger section.

### Human Inputs Required

- Click or approve additional waypoints.
- Confirm which failure mode is most common: detection, control sign, too much command, too little command, wall collision, hole fall, or path geometry.

### Development Work

- Add waypoint advancement tuning.
- Add speed/command caps per section if needed.
- Add recovery behavior only if a repeatable failure demands it.

### Acceptance Criteria

- The ball completes one meaningful section more often than it fails.
- Failures are logged and classified.

## Stage 10 - Full Maze Attempt

### Goal

Attempt full start-to-finish traversal.

### Preconditions

- Live tracker is reliable.
- Axis mapping is trusted.
- Short-segment runner works.
- At least one larger section works.
- Path waypoints are verified.
- Servo limits remain safe.

### Human Inputs Required

- Full route waypoints.
- Manual ball reset plan.
- Physical supervision during run.

### Acceptance Criteria

- The run either reaches finish or fails in a logged, explainable way.
- No uncontrolled servo motion.

## Explicit Non-Goals Until Later

- Do not use RL.
- Do not optimize full maze before a short segment works.
- Do not depend on side-mounted ChArUco for ball-plane calibration.
- Do not use screen recordings for tracker calibration.
- Do not widen servo limits because tracking/control is bad.
- Do not silently replace required human measurements with guessed constants.

## Prompt Template For A New Chat

Use this template for each stage:

```text
Read AGENTS.md and docs/LIVE_MVP_DEVELOPMENT_ROADMAP.md.
We are working on Stage <N>: <stage name>.
Do not implement yet. First inspect the relevant files listed in the stage, then ask for any human inputs required by that stage.
After the human provides those inputs, create a concrete implementation and validation plan for only this stage.
Then review whether the plan satisfies the acceptance criteria before executing.
```

When the plan is approved, use:

```text
Execute the approved Stage <N> plan.
Keep edits scoped to this stage.
Run tests or compile checks where possible.
Provide exact hardware validation steps for me to run.
Add an agent log in logs/agent/.
Stop after this stage and report whether acceptance criteria passed.
```
