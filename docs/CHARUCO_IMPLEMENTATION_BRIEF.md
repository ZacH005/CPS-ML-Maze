# CharUco Implementation Brief

## Goal

Use the CharUco board as a startup calibration target to rectify the camera view and produce a stable image-to-maze homography. After calibration, live ball tracking should run on the rectified maze plane without requiring CharUco detection every frame.

## Current Intended Flow

1. Detect the CharUco board in the current camera view.
2. Match detected CharUco image points to known maze-plane millimeter coordinates.
3. Estimate and save the homography from camera pixels to maze coordinates.
4. Use the saved homography for live ball tracking and control.
5. Do not keep CharUco active in the runtime loop unless the camera or board position changes.

## Coordinate Frames

- Image frame: pixels, origin at the top-left of the camera image.
- Maze frame: millimeters, origin at the bottom-left of the internal maze edge as seen by the camera.
- CharUco board frame: local calibration target geometry only; it is not the same thing as the maze frame unless explicitly mapped.

## Maze Measurements

The maze measurements supplied for the control frame are:

- Maze width: 3220 mm
- Maze height: 2820 mm
- External border wall width: 10 mm
- Internal visible outer width: 3010 mm
- Internal visible outer height: 2620 mm

Maze corners in the chosen origin frame:

- Bottom-left maze corner: (0, 0) mm
- Bottom-right maze corner: (3220, 0) mm
- Top-right maze corner: (3220, 2820) mm
- Top-left maze corner: (0, 2820) mm

## CharUco Board Measurements

The printed CharUco board is 6 cm by 6 cm, so the board is 60 mm square. Using 6x6 ArUco markers. 

Board geometry used in code:

- Squares across: 5
- Squares down: 5
- Square length: 12 mm
- Marker length: 9 mm

This is consistent with a 60 mm total board size when using millimeters.

## CharUco Placement On The Maze

The CharUco board is mounted on the maze and is offset from the maze origin. Use these maze-plane coordinates for the four CharUco board corners:

- Bottom-left CharUco corner: (-53, 107) mm
- Top-left CharUco corner: (-53, 167) mm
- Top-right CharUco corner: (7, 167) mm
- Bottom-right CharUco corner: (7, 107) mm

Equivalent description in centimeters:

- Bottom-left CharUco corner: (-5.3, 10.7) cm
- Top-left CharUco corner: (-5.3, 16.7) cm
- Top-right CharUco corner: (0.7, 16.7) cm
- Bottom-right CharUco corner: (0.7, 10.7) cm

Interpretation:

- The CharUco board extends 5.3 cm left of the maze origin.
- The CharUco board extends 0.7 cm right of the maze origin.
- The CharUco board extends from 10.7 cm to 16.7 cm above the maze origin.
- The board overlaps the maze edge slightly on the right side by about 0.7 cm.

## How To Use The CharUco Board

The CharUco board is useful for calibration-time geometry.

It should be used to:

- rectify the camera at startup
- compute the image-to-maze homography
- stabilize the maze into a flat planar coordinate frame for accurate ball tracking later

It should not be used as a live runtime dependency once the homography is saved.

## What The Implementation Needs

1. The CharUco detector in `src/cps_maze/vision/aruco.py` should use the real board geometry in millimeters.
2. The calibration step should map detected CharUco image points to the maze coordinates listed above.
3. The saved homography should represent the maze plane, not just the local CharUco board.
4. The live tracker should use the saved homography and operate in maze millimeters.

## Important Constraints

- The homography is tied to the current camera-and-board setup. If the camera moves, recalibrate.
- The CharUco board does not automatically define the maze edges; its placement on the maze must be measured and encoded once.
- The runtime ball tracker should not need CharUco detection if the calibration is already correct.

## Practical Notes For The Agent

- Keep all calibration units consistent in millimeters.
- Do not mix CharUco-local units with maze-plane units in the same transform.
- Use the CharUco board only as a startup calibration aid, then switch to ball tracking on the rectified maze plane.
- The control origin is the bottom-left of the internal maze edge as seen by the camera.