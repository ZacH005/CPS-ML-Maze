# Engineering Struggles and How They Were Resolved

Companion to the software setup document. Each entry is a real problem hit
during development, its root cause, and the resolution. These map directly to
"design iterations", "lessons learned", and "limitations" material for the
paper.

## 1. Hardware and electrical

### Burned power wiring from a stalled servo
The servo power rail was first wired with thin breadboard jumper wire. During
an actuation test the firmware travel limits had been removed to gain range;
a servo was commanded past the mechanism's end stop, could not reach its
setpoint, and drew sustained stall current. The undersized conductors
overheated to failure. Root cause was the combination of unconstrained travel
and wiring not rated for stall current. Resolution: the firmware travel
limits were restored permanently and the extra range was obtained
mechanically through the linkage instead. Lesson: never widen actuator limits
to compensate for a mechanical shortcoming, and treat stall current as the
design current for power wiring.

### First linkage had too little travel
The first coupling screwed a circular servo horn directly to the board. The
servo's usable swing through the firmware's safe pulse range produced too
little board rotation. Resolution: a push-rod linkage (straight horn, ball
joint rod ends on a threaded rod, wooden lever on the board) whose lever
ratios provide the needed range while the firmware limits stay conservative.

### Static friction dominates at small tilts
Below a measurable tilt the ball simply does not move. Early axis
calibration pulses at low amplitude produced zero displacement and useless
measurements. The calibration tool had to escalate pulse amplitude per axis
until the ball demonstrably moved, and the controller later needed explicit
stiction compensation (see controls section).

### Asymmetric and slack linkages
One axis initially moved the ball far less than the other (loose horn screw,
slack in rod ends), producing degenerate axis-map measurements. The servo
channels were also physically swapped relative to the firmware's assumption
at one point. Resolution: the axis mapping is measured, not assumed - a
calibration script pulses each servo axis and records the ball's response,
absorbing any swaps, sign flips, or asymmetries into a measured matrix.

## 2. Calibration

### Wrong board dimensions propagated silently
Three different play-area sizes circulated at different times (a mistyped
3220 x 2820, an assumed 322 x 282, and the final measured 263 x 222 mm).
Because the homography scales everything to the configured dimensions, every
derived artifact was silently wrong until re-measured with a ruler. Lesson:
physical dimensions are measurements, not config defaults, and every
downstream artifact must be regenerated when they change.

### ChArUco board mounted off the rolling plane
The first ChArUco calibration target was glued to a side platform next to
the board. A homography is only valid for points in a single plane, and the
pattern's plane was not the plane the ball rolls on, so the calibration was
systematically warped and initially unusable. Resolution: the pattern must
lie flat on the play surface during calibration; the corner-click
calibration served as the reliable default meanwhile.

### The camera is not a constant
The OS camera index changed between reboots and USB ports, so scripts
sometimes silently opened the laptop webcam or a virtual camera instead of
the maze camera. The default Windows capture backend also took tens of
seconds to open the camera, and the default uncompressed mode silently
capped the frame rate. Auto-exposure made the first frames of every capture
too dark for brightness-based seeding. Resolutions: per-machine config
overlay for the device index, DirectShow backend and MJPG mode on Windows,
and never seeding from the first frames of a stream.

## 3. Perception

### The ball looks like the holes
At the camera's resolution a reflective silver marble on a bright board is
nearly indistinguishable from the dark holes and their bright rims. A naive
bright-blob detector locked onto wall glints and hole rims; automatic
seeding regularly picked glare instead of the ball. The working detector
combines two cues that holes cannot satisfy simultaneously: motion
(frame-to-frame difference; holes do not move) and specular highlight (the
metal glint saturates brighter than printed features), plus an offline
calibration that blacklists locations that are bright too often across a
recording (hole rims, fixed glare) and a region-of-interest polygon that
excludes everything off the playable surface. For demos, seeding is manual:
the operator clicks the ball. Lesson: on this kind of scene, tracking
reliability came from cue combination and precomputed scene knowledge, not
from tuning a single threshold.

### Stationary balls disappear
Motion-based detection produces no signal when the ball stops, which is
exactly when the controller most needs feedback (stall detection). Gaps are
bridged by the highlight cue and short constant-velocity prediction, and
lighting must keep the glint above the specular threshold, which varied
between lab sessions and needed re-measurement.

## 4. Planning and control

### Pure position control parks the ball short
With proportional-derivative position control, the commanded tilt shrinks as
the ball approaches its target and falls below the tilt needed to overcome
static friction: the system reaches a stable equilibrium with the ball
parked short of the target ("balanced but not moving", observed for seconds
at a time in run logs). Resolution: explicit stiction compensation (a
minimum command magnitude once a commanded-but-not-moving state persists)
plus a small integral term.

### The stiction fix then caused corner crashes
The first stiction kick triggered on any single slow frame, but the ball is
also intentionally slow while braking into corners; the kick punched full
breakaway tilt into deliberate slowdowns and flung the ball into walls and
holes. Resolution: the kick requires the low-speed condition to persist
(distinguishing real static friction from intentional braking), and all
commands pass through a slew-rate limiter before reaching the hardware.

### Safety caps can silently disable safety fixes
A run with the command cap set below the stiction kick clipped the kick away
entirely: the ball received a command it physically could not respond to,
for the whole run, with no error anywhere. A loud warning now fires when the
configuration is self-defeating. Lesson: interacting safety mechanisms need
explicit consistency checks.

### The maze's geometry defeats naive path following
The route snakes, so corridors that are far apart along the route sit
millimetres apart behind a single wall. Projecting the ball onto the nearest
route segment sometimes associated it with a corridor on the other side of a
wall, and the controller then drove the ball into that wall. The first fix
(only allowing the association to move within a window of path progress per
frame) still failed inside chicanes, where the adjacent corridor is within
the window. The robust fix required wall knowledge: the walls are rasterized
once into a static obstacle mask, candidate associations whose line of sight
from the ball crosses a wall are rejected, and wall proximity imposes a
speed limit.

### Chicanes read as straight lines
Corner slowdown initially measured path curvature as the angle between the
tangent here and the tangent a fixed distance ahead. A chicane turns right
then left, the two turns cancel at the endpoints, and the measure reported
"straight" - so the ball entered chicanes at full cruise speed and
ricocheted between the walls. Resolution: curvature is accumulated absolute
turning along the span, which cannot cancel.

## 5. Process and collaboration

### Two operating systems, one repo
The team develops on macOS and Windows simultaneously. Serial ports
(/dev/cu.* vs COM), camera indices, python vs python3, and shell syntax
(bash line continuations vs PowerShell) all differ, and shared config edits
kept overwriting each machine's settings. Resolution: a gitignored
per-machine configuration overlay for device-specific values, and
documentation written for both shells.

### Parallel work on shared artifacts
Calibration artifacts and annotation CSVs were regenerated independently on
different machines, causing merge conflicts and stale-artifact confusion
(one machine's homography with another machine's path file silently
misaligns everything). Resolution: an explicit dependency rule - a new
homography invalidates and requires regenerating every derived artifact -
and backups of replaced artifacts committed alongside regenerated ones.

### Debugging by measurement, not impression
Early tuning was driven by watching the ball and guessing. Progress
accelerated after every run began writing a full log and an analysis tool
reported detection rate, progress reached, cross-track error statistics, and
stall episodes located by path position. Several supposed "controller
problems" turned out to be a too-low command cap, a wrong corridor
association, or a tracking dropout - all visible in the logs and invisible
to the eye.
