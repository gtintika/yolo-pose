import argparse
import time
from typing import List, Optional

import cv2
import numpy as np

from pose_detection import CleanPoseDetector, DEBUG, KEYPOINT_CONF_THRESHOLD


def parse_analyze_arg(analyze: str) -> Optional[List[str]]:
    if not analyze:
        return None
    cleaned = analyze.strip("{}")
    motions = [m.strip() for m in cleaned.split(",") if m.strip()]
    return motions or None


def depth_to_colormap(depth: Optional[np.ndarray], target_size: tuple[int, int]) -> np.ndarray:
    """Convert raw depth map to a viewable BGR image."""
    if depth is None:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    if depth.dtype != np.uint16 and depth.dtype != np.float32:
        depth = depth.astype(np.uint16)

    valid = depth > 0
    if not np.any(valid):
        depth_vis = np.zeros_like(depth, dtype=np.uint8)
    else:
        valid_depth = depth[valid].astype(np.float32)
        d_min = np.percentile(valid_depth, 5)
        d_max = np.percentile(valid_depth, 95)
        if d_max <= d_min:
            d_max = d_min + 1.0
        clipped = np.clip(depth.astype(np.float32), d_min, d_max)
        normalized = ((clipped - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        depth_vis = normalized

    colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    if (colored.shape[1], colored.shape[0]) != target_size:
        colored = cv2.resize(colored, target_size, interpolation=cv2.INTER_NEAREST)
    return colored


def _annotate_motions(
    annotated_frame: np.ndarray,
    keypoints_list: list,
    detector: CleanPoseDetector,
    analyze_motions: Optional[List[str]],
    frame_count: int,
    entity_label: str = 'Person',
) -> int:
    """Draw motion analysis text on frame. Returns the final y_offset."""
    y_offset = 30
    for person_id, keypoints in enumerate(keypoints_list):
        if analyze_motions:
            for motion_type in analyze_motions:
                action, normalized, angle = detector.analyze_motion(keypoints, motion_type)
                if DEBUG:
                    if angle is not None:
                        print(
                            f"Frame {frame_count} - {entity_label} {person_id + 1} - "
                            f"{action}: {normalized:.2f} (angle: {angle:.0f}deg)"
                        )
                    else:
                        print(f"Frame {frame_count} - {entity_label} {person_id + 1} - {action}: N/A")
                if angle is not None:
                    text = f"{entity_label[0]}{person_id + 1} {action}: {normalized:.2f}"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    y_offset += 25
    return y_offset


def _draw_wrist_coords(
    annotated_frame: np.ndarray,
    keypoints_list: list,
    depth_frame: Optional[np.ndarray] = None,
    y_offset: int = 30,
) -> int:
    """Draw left/right wrist x, y, z on frame. z comes from depth map if available."""
    h, w = annotated_frame.shape[:2]
    for person_id, keypoints in enumerate(keypoints_list):
        kp_by_name = {kp['name']: kp for kp in keypoints}
        for side in ('left', 'right'):
            wrist = kp_by_name.get(f'{side}_wrist')
            if wrist is None or wrist['confidence'] < 0.5:
                continue
            # Pixel coordinates
            px = int(wrist['x'] * w)
            py = int(wrist['y'] * h)
            # Depth lookup
            z_str = "N/A"
            if depth_frame is not None:
                dh, dw = depth_frame.shape[:2] if len(depth_frame.shape) >= 2 else (0, 0)
                if dh > 0 and dw > 0:
                    dx = int(wrist['x'] * dw)
                    dy = int(wrist['y'] * dh)
                    dx = min(max(dx, 0), dw - 1)
                    dy = min(max(dy, 0), dh - 1)
                    raw = depth_frame[dy, dx]
                    if len(depth_frame.shape) == 3:
                        raw = raw[0]
                    z_val = int(raw)
                    z_str = f"{z_val}mm" if z_val > 0 else "N/A"
            label = f"P{person_id + 1} {side}_wrist: x={px} y={py} z={z_str}"
            # On-frame overlay
            cv2.putText(
                annotated_frame, label, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2,
            )
            y_offset += 22
            # Draw small circle at wrist position
            cv2.circle(annotated_frame, (px, py), 5, (255, 255, 0), -1)
    return y_offset


def _add_info_text(frame: np.ndarray, frame_count: int, num_persons: int, entity_label: str = 'Person') -> None:
    """Add frame info text at the bottom of the frame."""
    height = frame.shape[0]
    info_text = f"Frame: {frame_count} | {entity_label}s: {num_persons}"
    cv2.putText(
        frame,
        info_text,
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def _print_stats(frame_count: int, inference_times: list, processing_times: list) -> None:
    if frame_count > 0:
        avg_inference = sum(inference_times) / len(inference_times) * 1000
        avg_processing = sum(processing_times) / len(processing_times) * 1000
        total_time = sum(processing_times)
        print("\n--- Stats ---")
        print(f"Frames processed: {frame_count}")
        print(f"Avg inference:    {avg_inference:.1f} ms/frame")
        print(f"Avg processing:   {avg_processing:.1f} ms/frame (inference + annotation + I/O)")
        print(f"Throughput:       {frame_count / total_time:.1f} FPS")
        print(f"Total time:       {total_time:.2f} s")


def _create_writer(
    output_path: Optional[str], width: int, height: int, fps: int, side_by_side: bool = False
) -> Optional[cv2.VideoWriter]:
    if not output_path:
        return None
    out_width = width * 2 if side_by_side else width
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))
    print(f"Recording to: {output_path}")
    return writer


def _draw_show_points(
    annotated_frame: np.ndarray,
    keypoints_list: list,
    show_points: List[str],
    width: int,
    height: int,
    frame_count: int = 0,
    entity_label: str = 'Person',
) -> None:
    """Overlay keypoint x,y coordinates on frame (max 3 keypoints)."""
    colors = [(0, 0, 139), (139, 0, 0), (0, 100, 0)]
    for person_id, keypoints in enumerate(keypoints_list):
        for color, kpt_name in zip(colors, show_points[:3]):
            kpt = next((k for k in keypoints if k['name'] == kpt_name), None)
            if kpt and kpt['confidence'] >= KEYPOINT_CONF_THRESHOLD:
                if DEBUG:
                    print(f"Frame {frame_count} - {entity_label} {person_id+1} - {kpt_name}: x={kpt['x']:.3f}, y={kpt['y']:.3f}, conf={kpt['confidence']:.2f}")
                px = int(kpt['x'] * width)
                py = int(kpt['y'] * height)
                label = f"{entity_label[0]}{person_id+1} {kpt_name}: ({kpt['x']:.3f}, {kpt['y']:.3f})"
                cv2.putText(annotated_frame, label, (px + 8, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.circle(annotated_frame, (px, py), 5, color, -1)


def process_camera_stream(
    detector: CleanPoseDetector,
    camera_index: int = 0,
    output_path: Optional[str] = None,
    show: bool = True,
    analyze_motions: Optional[List[str]] = None,
    show_points: Optional[List[str]] = None,
) -> None:
    """Process live Orbbec camera with color + depth side-by-side."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_OBSENSOR)
    if not cap.isOpened():
        print(
            "Error: Could not open Orbbec camera via OpenCV CAP_OBSENSOR.\n"
            "Make sure an Orbbec camera is connected and OpenCV was built with obsensor support."
        )
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"Orbbec stream: {width}x{height} @ {fps}fps")

    writer = _create_writer(output_path, width, height, fps, side_by_side=False)
    if writer:
        print(f"Recording annotated color stream to: {output_path}")

    print("Processing Orbbec color + depth... Press 'q' to quit")

    frame_count = 0
    inference_times = []
    processing_times = []

    try:
        while True:
            frame_start = time.perf_counter()

            if not cap.grab():
                print("End of stream or frame grab failed")
                break

            ok_color, color_frame = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
            ok_depth, depth_frame = cap.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)

            if not ok_color or color_frame is None:
                print("Warning: Color frame unavailable, skipping frame")
                continue

            annotated_frame, keypoints_list = detector.detect_pose(color_frame, normalize_coords=True)

            inference_time = time.perf_counter() - frame_start
            inference_times.append(inference_time)

            y_off = _annotate_motions(annotated_frame, keypoints_list, detector, analyze_motions, frame_count, detector.entity_label)
            _draw_wrist_coords(annotated_frame, keypoints_list, depth_frame if ok_depth else None, y_off)
            if show_points:
                _draw_show_points(annotated_frame, keypoints_list, show_points, width, height, frame_count, detector.entity_label)
            _add_info_text(annotated_frame, frame_count, len(keypoints_list), detector.entity_label)

            if writer:
                writer.write(annotated_frame)

            depth_vis = depth_to_colormap(depth_frame if ok_depth else None, (width, height))
            if not ok_depth:
                cv2.putText(
                    depth_vis,
                    "Depth stream unavailable",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            if show:
                preview = np.hstack((annotated_frame, depth_vis))
                cv2.imshow("Orbbec Pose Detection (Color | Depth)", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quitting...")
                    break

            processing_time = time.perf_counter() - frame_start
            processing_times.append(processing_time)
            frame_count += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        _print_stats(frame_count, inference_times, processing_times)


def process_file_stream(
    detector: CleanPoseDetector,
    source: str,
    stream_type: str = "color",
    depth_source: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = True,
    analyze_motions: Optional[List[str]] = None,
    show_points: Optional[List[str]] = None,
) -> None:
    """Process video file(s). Supports color, depth, or side-by-side modes."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {source}")
        return

    depth_cap = None
    if depth_source:
        depth_cap = cv2.VideoCapture(depth_source)
        if not depth_cap.isOpened():
            print(f"Error: Could not open depth video file: {depth_source}")
            cap.release()
            return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"Video source: {source} ({width}x{height} @ {fps}fps)")
    if depth_source:
        print(f"Depth source: {depth_source}")

    side_by_side = depth_cap is not None
    writer = _create_writer(output_path, width, height, fps, side_by_side=side_by_side)

    mode_desc = "color + depth side-by-side" if side_by_side else stream_type
    print(f"Processing {mode_desc}... Press 'q' to quit")

    frame_count = 0
    inference_times = []
    processing_times = []

    try:
        while True:
            frame_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok or frame is None:
                print("End of video")
                break

            depth_frame = None
            if depth_cap is not None:
                ok_depth, depth_frame = depth_cap.read()
                if not ok_depth:
                    print("End of depth video")
                    break

            if side_by_side:
                # Color file + depth file: pose on color, colormap on depth
                annotated_frame, keypoints_list = detector.detect_pose(frame, normalize_coords=True)
                inference_times.append(time.perf_counter() - frame_start)
                y_off = _annotate_motions(annotated_frame, keypoints_list, detector, analyze_motions, frame_count, detector.entity_label)
                _draw_wrist_coords(annotated_frame, keypoints_list, depth_frame, y_off)
                if show_points:
                    _draw_show_points(annotated_frame, keypoints_list, show_points, width, height, frame_count, detector.entity_label)
                _add_info_text(annotated_frame, frame_count, len(keypoints_list), detector.entity_label)
                depth_vis = depth_to_colormap(depth_frame, (width, height))
                display = np.hstack((annotated_frame, depth_vis))
            elif stream_type == "depth":
                # Depth file only: colorize, no pose detection
                display = depth_to_colormap(frame, (width, height))
                inference_times.append(time.perf_counter() - frame_start)
            else:
                # Color file only: pose detection (no depth available)
                annotated_frame, keypoints_list = detector.detect_pose(frame, normalize_coords=True)
                inference_times.append(time.perf_counter() - frame_start)
                y_off = _annotate_motions(annotated_frame, keypoints_list, detector, analyze_motions, frame_count, detector.entity_label)
                _draw_wrist_coords(annotated_frame, keypoints_list, None, y_off)
                if show_points:
                    _draw_show_points(annotated_frame, keypoints_list, show_points, width, height, frame_count, detector.entity_label)
                _add_info_text(annotated_frame, frame_count, len(keypoints_list), detector.entity_label)
                display = annotated_frame

            if writer:
                writer.write(display)

            if show:
                window_name = "Orbbec Pose Detection"
                if side_by_side:
                    window_name += " (Color | Depth)"
                elif stream_type == "depth":
                    window_name += " (Depth)"
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quitting...")
                    break

            processing_time = time.perf_counter() - frame_start
            processing_times.append(processing_time)
            frame_count += 1
    finally:
        cap.release()
        if depth_cap is not None:
            depth_cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        _print_stats(frame_count, inference_times, processing_times)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pose detection using Orbbec camera or recorded .mov files"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="camera",
        help="Video file path or 'camera' for live Orbbec camera (default: camera)",
    )
    parser.add_argument(
        "--stream-type",
        type=str,
        default="color",
        choices=["color", "depth"],
        help="Stream type when using a file source: 'color' runs pose detection, 'depth' shows colorized depth (default: color)",
    )
    parser.add_argument(
        "--depth-source",
        type=str,
        default=None,
        help="Optional depth video file to display side-by-side with color source",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Orbbec camera index (for live camera mode)")
    parser.add_argument("--model", type=str, default="yolo26m-pose.pt", help="YOLO model name")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Output path for recorded video")
    parser.add_argument("--no-show", action="store_true", help="Do not display preview windows")
    parser.add_argument(
        "--analyze",
        type=str,
        default="",
        help="Motions to analyze as comma-separated names in braces. "
             "Available: right_arm_abduction, left_arm_abduction, "
             "right_elbow_flexion, left_elbow_flexion, "
             "right_knee_flexion, left_knee_flexion, "
             "right_hand_raise, left_hand_raise. "
             "Example: {right_arm_abduction,left_hand_raise}",
    )
    parser.add_argument(
        "--show-points",
        type=str,
        default="",
        help="Keypoint names to overlay x,y on frame (max 3), e.g., {left_wrist,right_wrist}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference: cpu, mps, cuda, or cuda:0 (default: auto)",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate detections (disable filtering)",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.7,
        help="Keypoint similarity threshold for duplicates (0-1, lower = more aggressive)",
    )

    args = parser.parse_args()

    detector = CleanPoseDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        remove_duplicates=not args.keep_duplicates,
        duplicate_threshold=args.duplicate_threshold,
        device=args.device,
    )

    analyze_motions = parse_analyze_arg(args.analyze)

    show_points = None
    if args.show_points:
        cleaned = args.show_points.strip('{}')
        show_points = [p.strip() for p in cleaned.split(',') if p.strip()][:3]

    if args.source == "camera":
        process_camera_stream(
            detector=detector,
            camera_index=args.camera_index,
            output_path=args.output,
            show=not args.no_show,
            analyze_motions=analyze_motions,
            show_points=show_points,
        )
    else:
        process_file_stream(
            detector=detector,
            source=args.source,
            stream_type=args.stream_type,
            depth_source=args.depth_source,
            output_path=args.output,
            show=not args.no_show,
            analyze_motions=analyze_motions,
            show_points=show_points,
        )


if __name__ == "__main__":
    main()
