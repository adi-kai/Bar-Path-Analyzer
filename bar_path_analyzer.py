import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys


#Color ranges in HSV for common bumper plate colors
COLOR_RANGES = {
    "red": [
    (np.array([0, 120, 70]),   np.array([10, 255, 255])),
    (np.array([170, 120, 70]), np.array([180, 255, 255])),
    # Dark maroon/Eleiko range
    (np.array([0, 60, 30]),    np.array([15, 200, 140])),
    (np.array([165, 60, 30]),  np.array([180, 200, 140])),
    ],
    "blue": [
        (np.array([100, 150, 50]), np.array([130, 255, 255])),
    ],
   "black": [
    (np.array([0, 0, 20]),  np.array([180, 255, 130])),
    ],
    "green": [
    (np.array([35, 40, 40]),  np.array([90, 255, 255])),
    ],
    "yellow": [
        (np.array([20, 100, 100]), np.array([35, 255, 255])),
    ],
}

# Velocity colormap: slow (blue) → fast (red)
VELOCITY_CMAP = [
    (0.0,  (255, 50,  50)),
    (0.25, (50,  200, 50)),
    (0.5,  (50,  255, 200)),
    (0.75, (50,  200, 255)),
    (1.0,  (50,  50,  255)),
]


def lerp_color(t: float) -> tuple:
    """Interpolate along VELOCITY_CMAP for a normalized speed t ∈ [0, 1]."""
    for i in range(len(VELOCITY_CMAP) - 1):
        t0, c0 = VELOCITY_CMAP[i]
        t1, c1 = VELOCITY_CMAP[i + 1]
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0)
            return tuple(int(c0[j] + a * (c1[j] - c0[j])) for j in range(3))
    return VELOCITY_CMAP[-1][1]


def detect_bar_center(frame: np.ndarray, color_key: str = "auto") -> tuple | None:
    """
    Detect the center of the barbell plate in a frame.
    Returns (cx, cy) in pixel coords, or None if not found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]


    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(roi_mask, (w // 6, 0), (5 * w // 6, int(h * 0.85)), 255, -1)
    keys = list(COLOR_RANGES.keys()) if color_key == "auto" else [color_key]

    best = None
    best_area = 0

    for key in keys:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in COLOR_RANGES[key]:
            mask |= cv2.inRange(hsv, lo, hi)

        # Apply ROI mask
        mask = cv2.bitwise_and(mask, roi_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 400:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.55:  # FIXED: skip non-circular shapes
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            if not (0.02 * w < 2 * radius < 0.35 * w):
                continue

            if area > best_area:
                best_area = area
                best = (int(x), int(y))

    return best


def velocity_from_positions(positions: list, fps: float) -> list:
    """Compute instantaneous pixel/frame velocity between consecutive positions."""
    vels = [0.0]
    for i in range(1, len(positions)):
        if positions[i] and positions[i - 1]:
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            vels.append(np.sqrt(dx**2 + dy**2) * fps)
        else:
            vels.append(0.0)
    return vels


def draw_bar_path(
    canvas: np.ndarray,
    positions: list,
    velocities: list,
    max_vel: float,
    thickness: int = 8,
) -> None:
    """Draw the color-coded bar path trail onto canvas in-place."""
    for i in range(1, len(positions)):
        if positions[i] is None or positions[i - 1] is None:
            continue
        t = velocities[i] / max_vel if max_vel > 0 else 0
       
        color = lerp_color(t)  
        cv2.line(canvas, positions[i - 1], positions[i], color, thickness, cv2.LINE_AA)

def draw_skeleton(frame: np.ndarray, landmarks, connections, color=(200, 200, 200)) -> None:
    """Draw pose skeleton lines and joint dots onto frame in-place."""
    h, w = frame.shape[:2]

    for (a, b) in connections:
        lm_a = landmarks[a]
        lm_b = landmarks[b]
        if lm_a.visibility > 0.4 and lm_b.visibility > 0.4:
            xa, ya = int(lm_a.x * w), int(lm_a.y * h)
            xb, yb = int(lm_b.x * w), int(lm_b.y * h)
            cv2.line(frame, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)

    for lm in landmarks:
        if lm.visibility > 0.4:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, color, 1, cv2.LINE_AA)


def annotate_stats(
    frame: np.ndarray,
    peak_height_m: float,
    peak_vel_ms: float,
    peak_height_px: tuple | None,
    peak_vel_px: tuple | None,
) -> None:
    """Draw annotation boxes for peak height and peak velocity."""
    font = cv2.FONT_HERSHEY_DUPLEX
    box_color = (40, 40, 40)
    alpha = 0.72

    def draw_box(pos, lines, dot_color):
        if pos is None:
            return
        x, y = pos
        line_h = 28
        pad = 10
        max_w = max(cv2.getTextSize(l, font, 0.55, 1)[0][0] for l in lines)
        bx, by = x + 14, y - len(lines) * line_h // 2
        bw, bh = max_w + pad * 2, len(lines) * line_h + pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), box_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (180, 180, 180), 1)

        for i, line in enumerate(lines):
            cv2.putText(
                frame, line, (bx + pad, by + pad + (i + 1) * line_h - 6),
                font, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
            )
        cv2.circle(frame, pos, 8, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, pos, 8, (255, 255, 255), 2, cv2.LINE_AA)

    draw_box(
        peak_height_px,
        ["Peak Height", f"{peak_height_m:.2f} m"],
        (255, 220, 50),
    )
    draw_box(
        peak_vel_px,
        ["Peak Velocity", f"{peak_vel_ms:.2f} m/s"],
        (50, 200, 255),
    )


def add_title_bar(frame: np.ndarray, text: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 52), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(
        frame, text, (16, 35),
        cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA,
    )


def add_legend(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    bar_w, bar_h = 120, 10
    x0, y0 = 14, h - 30

    for i in range(bar_w):
        t = i / bar_w
        color = lerp_color(t)
        cv2.line(frame, (x0 + i, y0), (x0 + i, y0 + bar_h), color, 1)

    cv2.putText(frame, "slow", (x0, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "fast", (x0 + bar_w - 22, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "color = bar speed", (x0, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)


def process_video(
    input_path: str,
    output_path: str,
    bar_color: str = "auto",
    draw_skel: bool = True,
    show: bool = False,
    title: str = "Bar Path + Peak Velocity Analysis",
    max_jump_px: int = 80, 
) -> None:

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if VideoWriter can be initialized successfully
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[ERROR] Could not open video writer for output path: {output_path}")
        print("        Try using .avi extension on Windows or ensure codecs are installed.")
        sys.exit(1)

    # MediaPipe setup 
    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    CONNECTIONS = mp_pose.POSE_CONNECTIONS

    # Pass 1: collect bar positions across all frames
    print(f"[1/2] Detecting bar positions across {n_frames} frames...")
    all_positions = []
    all_frames    = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pos = detect_bar_center(frame, bar_color)

        # Reject detections that teleport (almost always false positives)
        if pos is not None and all_positions and all_positions[-1] is not None:
            prev = all_positions[-1]
            dist = ((pos[0] - prev[0])**2 + (pos[1] - prev[1])**2) ** 0.5
            if dist > max_jump_px: # Used max_jump_px parameter
                pos = None

        all_positions.append(pos)
        all_frames.append(frame.copy())
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  ... {frame_idx}/{n_frames}", end="\r")

    cap.release()
    print(f"\n  Done. Bar detected in {sum(1 for p in all_positions if p)} / {len(all_positions)} frames.")

    # Pixel -> meter calibration
    floor_y = height 
    valid_y_positions = [p[1] for p in all_positions if p is not None]
    if valid_y_positions:
        floor_y = max(valid_y_positions)

    PX_PER_METER = height / 2.2  # fallback: 2.2 m covers floor-to-overhead

    # Velocity calculation
    raw_vels_px   = velocity_from_positions(all_positions, fps)
    velocities_ms = [v / PX_PER_METER for v in raw_vels_px]

   
    nonzero_vels_ms = [v for v in velocities_ms if v > 0]
    max_vel_ms = np.percentile(nonzero_vels_ms, 95) if nonzero_vels_ms else 1.0

    nonzero_vels_px = [v for v in raw_vels_px if v > 0]
    max_vel_px = np.percentile(nonzero_vels_px, 95) if nonzero_vels_px else 1.0

    # Peak stats
    valid = [(i, p) for i, p in enumerate(all_positions) if p is not None]
    if valid:
        # Peak height measured from floor_y
        peak_h_idx, peak_h_pos = min(valid, key=lambda x: x[1][1])
        peak_h_m = (floor_y - peak_h_pos[1]) / PX_PER_METER

        peak_v_idx = int(np.argmax(velocities_ms))
        peak_v_pos = all_positions[peak_v_idx]
        peak_v_ms  = velocities_ms[peak_v_idx]
    else:
        peak_h_idx, peak_h_pos, peak_h_m = 0, None, 0.0
        peak_v_idx, peak_v_pos, peak_v_ms = 0, None, 0.0

    print(f"  Peak height : {peak_h_m:.2f} m  |  Peak velocity: {peak_v_ms:.2f} m/s")
    
    print(f"  max_vel_px: {max_vel_px}")
    print(f"  max_vel_ms: {max_vel_ms}")
    print(f"  nonzero vel count: {len(nonzero_vels_px)}")
    print(f"  sample positions (first 10): {all_positions[:10]}")
    # Pass 2: render annotated video
    print("[2/2] Rendering annotated video...")

    trail_len = max(1, int(fps * 10))

    for i, frame in enumerate(all_frames):
        frame_copy = frame.copy() # Bug fixed: Draw on a copy to prevent mutation

        if draw_skel:
            rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                draw_skeleton(frame_copy, results.pose_landmarks.landmark, CONNECTIONS)

        trail_start = max(0, i - trail_len)
        trail_pos   = all_positions[trail_start : i + 1]
        trail_vels  = raw_vels_px[trail_start : i + 1]
        draw_bar_path(frame_copy, trail_pos, trail_vels, max_vel_px)

        ph_px = peak_h_pos if i >= peak_h_idx else None
        pv_px = peak_v_pos if i >= peak_v_idx else None
        annotate_stats(frame_copy, peak_h_m, peak_v_ms, ph_px, pv_px)

        if all_positions[i]:
            cv2.circle(frame_copy, all_positions[i], 7, (255, 80, 80), -1, cv2.LINE_AA)
            cv2.circle(frame_copy, all_positions[i], 7, (255, 255, 255), 2, cv2.LINE_AA)

        add_title_bar(frame_copy, title)
        add_legend(frame_copy)

        writer.write(frame_copy)

        if show:
            cv2.imshow("Bar Path Analyzer", frame_copy)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if i % 30 == 0:
            print(f"  ... {i}/{len(all_frames)}", end="\r")

    writer.release()
    if show:
        cv2.destroyAllWindows()
    pose.close()

    print(f"\n[✓] Output saved to: {output_path}")


# CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Barbell bar path analyzer")
    parser.add_argument("--input",       required=True,  help="Input video path")
    parser.add_argument("--output",      default="bar_path_output.mp4")
    parser.add_argument("--bar_color",   default="auto",
                        choices=["auto", "red", "blue", "black", "green", "yellow"])
    parser.add_argument("--show",        action="store_true", help="Live preview")
    parser.add_argument("--no_skeleton", action="store_true")
    parser.add_argument("--title",       default="Bar Path + Peak Velocity Analysis")
    parser.add_argument("--max_jump",    type=int, default=80, help="Max pixels bar can move between frames (default: 80)") # Added CLI flag
    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        bar_color=args.bar_color,
        draw_skel=not args.no_skeleton,
        show=args.show,
        title=args.title,
        max_jump_px=args.max_jump, # Passed new argument
    )
