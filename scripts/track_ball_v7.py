"""
Real-time-capable ball tracker for the tilting labyrinth rig, using the
same gated-motion + specular-highlight method validated in autolabel.py
(94% precision on sampled frames) -- no trained model needed.

Why this approach and not a static per-frame detector: at this camera's
resolution the numbered holes are visually almost identical to the ball
(similar size, similar gray value). The one reliable discriminator is
motion (the ball moves, holes don't) combined with the ball's sharp,
near-saturated specular highlight (holes/text top out ~150-215 in
brightness, the ball hits ~254). Frame-to-frame (not fixed-background)
differencing keeps this robust even while the whole board tilts and its
shading shifts.

Limitations (inherent to a motion-based method, not a bug):
- Needs a previous frame -- this tracks video, it does not detect a
  ball in a single standalone photo.
- Cannot see the ball while it is perfectly still (zero motion -> zero
  diff signal). Gaps are bridged with constant-velocity prediction
  (marked "predicted" in the output) rather than left blank, since a
  ball rolling under gravity doesn't teleport between frames.

Two-step workflow (calibration vs. streaming inference):
    Static-confuser detection (compute_static_confusers) needs to scan
    hundreds of frames spread across an entire recording to tell "always
    a bit bright" apart from "the ball, passing through" -- that's only
    possible with a complete recorded video, never with a live stream.
    So it's a separate, offline, run-once step; tracking itself only
    ever looks at the current + previous frame and loads the saved
    result, which is exactly what a live feed can do too.

    # 1) calibrate once against footage you already have in full
    python track_ball.py --calibrate video.mp4 --confusers-file ball_detect/confusers.json

    # 2) track (this path never scans ahead -- fine for a live stream)
    python track_ball.py video.mp4 --seed-x 584 --seed-y 58 \
        --confusers-file ball_detect/confusers.json \
        --out-video annotated.mp4 --out-csv track.csv

    # find a good seed automatically from frame 0 (brightest small blob)
    python track_ball.py video.mp4 --auto-seed --out-csv track.csv
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def motion_candidates(prev_gray, next_gray, min_area=50, max_area=700, min_circularity=0.5):
    diff = cv2.GaussianBlur(cv2.absdiff(prev_gray, next_gray), (5, 5), 0)
    _, mask = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < min_circularity:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        out.append((x, y, r))
    return out


def highlight_candidates(gray, thresh=225, min_area=1, max_area=60, ball_r=9):
    """
    Direct per-frame appearance cue: connected components of near-saturated
    pixels (the ball's specular highlight). Unlike motion_candidates this
    works even when the ball is stationary or motion-blurred, at the cost
    of also picking up other shiny objects (screws, tools) elsewhere in
    frame -- callers must gate by proximity to a predicted position.
    """
    mask = (gray >= thresh).astype(np.uint8)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        cx, cy = centroids[i]
        out.append((float(cx), float(cy), float(ball_r)))
    return out


def specular_peak(gray, x, y, r):
    h, w = gray.shape
    x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
    y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
    patch = gray[y0:y1, x0:x1]
    return int(patch.max()) if patch.size else 0


def auto_seed(gray, min_specular=225):
    """Brightest small blob in the frame -- good enough for a static start position."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, maxval, _, maxloc = cv2.minMaxLoc(blurred)
    if maxval < min_specular:
        return None
    return float(maxloc[0]), float(maxloc[1])


def compute_static_confusers(video_path, n_samples=200, thresh=225, freq_thresh=0.10, margin=18):
    """
    Find every board location that is bright enough to pass the ball's
    specular test *suspiciously often across the video* -- these are
    static reflective features (peg/hole rims), not the ball.

    A first version of this used a single median-background frame, which
    only catches confusers that are bright in *most* frames. That missed
    a hole whose rim only glints ~35% of the time (enough to sustain a
    338-frame false lock once the tracker's prediction settled on it) --
    a true median blurs an intermittent glint below the threshold. Instead
    we count, per pixel, what *fraction* of sampled frames cross the
    specular threshold there. The real ball, which must travel across the
    whole board over the video, cannot linger on any single pixel for more
    than a few percent of frames -- verified: ~4% at its own start point,
    vs 35-93% for the two confusers found so far. A frequency threshold
    cleanly separates the two.
    """
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, n - 1), n_samples).astype(int)

    freq = None
    count = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        bright = (gray >= thresh)
        freq = bright.astype(np.uint32) if freq is None else freq + bright
        count += 1
    cap.release()
    if freq is None or count == 0:
        return []

    freq_map = (freq / count) >= freq_thresh
    mask = (freq_map.astype(np.uint8)) * 255
    n_comp, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Note: several adjacent confusers can merge into one much larger blob
    # (verified: a cluster of nearby holes on a different camera angle
    # merged into a single 39752px component). An earlier version capped
    # accepted component area at 200px to avoid ever excluding something
    # board-sized -- that silently threw out the *whole* merged cluster,
    # excluding none of it. Size from the bounding-box diagonal (correct
    # for large/irregular merged shapes, unlike area-derived radius) and
    # only cap the resulting radius, so oversized clusters still get
    # excluded instead of ignored outright.
    max_r = 70
    out = []
    for i in range(1, n_comp):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        r = min(0.5 * float(np.hypot(w, h)), max_r)
        out.append((float(cx), float(cy), r + margin))
    return out


def save_calibration(confusers, roi, path):
    Path(path).write_text(json.dumps({"confusers": confusers, "roi": roi}))


def load_calibration(path):
    data = json.loads(Path(path).read_text())
    return [tuple(c) for c in data["confusers"]], data.get("roi")


def in_roi(x, y, roi):
    """roi is a polygon (list of [x,y]) hugging the playable board surface --
    anything outside it (desk clutter, screws, cables) is never a candidate,
    regardless of how bright or how well it moves."""
    if not roi:
        return True
    poly = np.array(roi, dtype=np.float32)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


class BallTracker:
    """Feed grayscale frames one at a time via update(); works for live streams too."""

    def __init__(self, seed_xy, seed_r=9, max_jump=35, max_search=60,
                 search_growth=1.15, min_specular=225, max_predict_frames=8,
                 static_confusers=None, roi=None):
        self.pos = np.array(seed_xy, dtype=float)
        self.vel = np.zeros(2)
        self.r = seed_r
        self.max_jump = max_jump
        self.max_search = max_search
        self.search_growth = search_growth
        self.min_specular = min_specular
        self.max_predict_frames = max_predict_frames
        self.miss_streak = 0
        self.prev_gray = None
        # Some static board features (e.g. a raised peg's rim) are bright
        # enough to pass the specular test as reliably as the ball itself
        # (verified: hole "5" hits it in 98% of sampled frames). These are
        # precomputed once from a ball-free background (see
        # compute_static_confusers) and permanently excluded -- this is
        # what actually tells "always-bright board feature" apart from
        # "the ball, currently sitting still", which a purely reactive
        # stillness check cannot.
        self.static_confusers = static_confusers or []
        # Anything outside the playable board surface (desk clutter, the
        # screw pile, cables) should never be reported as the ball, no
        # matter how bright or how well it happens to move.
        self.roi = roi

    def _filter_candidates(self, cands):
        out = []
        for c in cands:
            x, y = c[0], c[1]
            if not in_roi(x, y, self.roi):
                continue
            if any(np.hypot(x - sx, y - sy) <= sr for (sx, sy, sr) in self.static_confusers):
                continue
            out.append(c)
        return out

    def update(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            return self.pos[0], self.pos[1], self.r, "seed"

        # union of two independent cues: motion (works on fast-moving ball,
        # ignored by static holes) and raw appearance (works on a stationary
        # or slow ball, where motion diff has nothing to show)
        cands = motion_candidates(self.prev_gray, gray) + highlight_candidates(gray, ball_r=self.r)
        cands = self._filter_candidates(cands)

        if self.miss_streak <= self.max_predict_frames:
            search_r = min(self.max_jump * (self.search_growth ** self.miss_streak), self.max_search)
            predicted = self.pos + self.vel
            best, best_d = None, None
            for (x, y, r) in cands:
                d = np.hypot(x - predicted[0], y - predicted[1])
                if d > search_r or specular_peak(gray, x, y, r) < self.min_specular:
                    continue
                if best_d is None or d < best_d:
                    best, best_d = (x, y, r), d
        else:
            # long lost: reacquire anywhere via strongest specular hotspot
            best, best_peak = None, -1
            for (x, y, r) in cands:
                peak = specular_peak(gray, x, y, r)
                if peak >= self.min_specular and peak > best_peak:
                    best, best_peak = (x, y, r), peak

        self.prev_gray = gray

        if best is not None:
            new_pos = np.array([best[0], best[1]])
            self.vel = new_pos - self.pos if self.miss_streak == 0 else (new_pos - self.pos) / (self.miss_streak + 1)
            self.pos = new_pos
            self.r = 0.7 * self.r + 0.3 * best[2]
            self.miss_streak = 0
            return self.pos[0], self.pos[1], self.r, "detected"

        self.miss_streak += 1
        if self.miss_streak <= self.max_predict_frames:
            self.pos = self.pos + self.vel  # constant-velocity coast through the gap
            return self.pos[0], self.pos[1], self.r, "predicted"

        return self.pos[0], self.pos[1], self.r, "lost"


def main():
    ap = argparse.ArgumentParser(description="Track the metallic ball through rig video.")
    ap.add_argument("video", nargs="?", help="video to track (not required for --calibrate)")
    ap.add_argument("--seed-x", type=float)
    ap.add_argument("--seed-y", type=float)
    ap.add_argument("--seed-r", type=float, default=9)
    ap.add_argument("--auto-seed", action="store_true", help="find seed from frame 0 automatically")
    ap.add_argument("--out-video", default=None, help="optional annotated output video")
    ap.add_argument("--out-csv", default="track.csv")
    ap.add_argument("--end-frame", type=int, default=None,
                     help="stop before this frame index (use to cut trailing non-maze footage)")
    ap.add_argument("--start-frame", type=int, default=0,
                     help="seed position is read from this frame (use when the ball isn't visible at frame 0)")
    ap.add_argument("--calibrate", metavar="SOURCE_VIDEO",
                     help="offline step: scan a full recorded video for static confusers "
                          "(holes/pegs that glint like the ball) and save them -- run this "
                          "once against footage you already have in full. Not usable on a "
                          "live stream, which by definition has no 'whole video' to scan.")
    ap.add_argument("--confusers-file", default="ball_detect/confusers.json",
                     help="where --calibrate saves to / normal tracking loads from")
    ap.add_argument("--roi", default=None,
                     help="playable-board polygon as 'x1,y1;x2,y2;...' (only used with --calibrate); "
                          "candidates outside it are never the ball, regardless of brightness/motion")
    args = ap.parse_args()

    if args.calibrate:
        print(f"scanning {args.calibrate} for static confusers (bright pegs/holes)...")
        confusers = compute_static_confusers(args.calibrate)
        roi = None
        if args.roi:
            roi = [[float(v) for v in pair.split(",")] for pair in args.roi.split(";")]
        save_calibration(confusers, roi, args.confusers_file)
        print(f"found {len(confusers)} confuser(s)" + (f", roi with {len(roi)} points" if roi else ", no roi given")
              + f", saved to {args.confusers_file}")
        return

    if not args.video:
        raise SystemExit("pass a video to track, or --calibrate SOURCE_VIDEO to (re)build confusers first")

    cap = cv2.VideoCapture(args.video)
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("could not read video")
    gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if args.auto_seed:
        seed = auto_seed(gray0)
        if seed is None:
            raise SystemExit("auto-seed failed: no bright specular blob in frame 0, pass --seed-x/--seed-y")
        print(f"auto-seed: {seed}")
    else:
        if args.seed_x is None or args.seed_y is None:
            raise SystemExit("pass --seed-x/--seed-y or --auto-seed")
        seed = (args.seed_x, args.seed_y)

    # Streaming-safe: this loads a file written once by a prior --calibrate
    # pass over footage we already had in full. It does NOT scan args.video
    # itself, so this same code path works unchanged on a live feed -- only
    # the calibration step needs a complete recording.
    confusers, roi = [], None
    if Path(args.confusers_file).exists():
        confusers, roi = load_calibration(args.confusers_file)
        print(f"loaded {len(confusers)} confuser(s)" + (f" and roi ({len(roi)} pts)" if roi else " (no roi)")
              + f" from {args.confusers_file}")
    else:
        print(f"no confusers file at {args.confusers_file} -- run with --calibrate first; "
              f"tracking without confuser/roi exclusion for now")

    tracker = BallTracker(seed, seed_r=args.seed_r, static_confusers=confusers, roi=roi)

    writer = None
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))

    rows = []
    counts = {"seed": 0, "detected": 0, "predicted": 0, "lost": 0}

    idx = args.start_frame
    while True:
        if args.end_frame is not None and idx >= args.end_frame:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, r, status = tracker.update(gray)
        counts[status] += 1
        rows.append((idx, round(x, 1), round(y, 1), round(r, 1), status))

        if writer is not None:
            color = {"seed": (255, 0, 0), "detected": (0, 255, 0),
                     "predicted": (0, 200, 255), "lost": (0, 0, 255)}[status]
            out = frame.copy()
            pad = r * 1.3
            x0, y0 = int(x - pad), int(y - pad)
            x1, y1 = int(x + pad), int(y + pad)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            cv2.putText(out, status, (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            writer.write(out)

        idx += 1
        ok, frame = cap.read()
        if not ok:
            break

    cap.release()
    if writer is not None:
        writer.release()

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "r", "status"])
        w.writerows(rows)

    total = sum(counts.values())
    print(f"frames: {total}")
    for k, v in counts.items():
        print(f"  {k}: {v} ({100*v/total:.1f}%)")
    print(f"wrote {args.out_csv}" + (f" and {args.out_video}" if args.out_video else ""))


if __name__ == "__main__":
    main()
