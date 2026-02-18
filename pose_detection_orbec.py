import argparse
import time
from typing import List, Optional

import cv2
import numpy as np

from pose_detection import (
    CleanPoseDetector,
    DEBUG,
    KEYPOINT_CONF_THRESHOLD,
    _annotate_motions,
    _draw_show_points,
    _add_info_text,
    _print_stats,
    build_base_parser,
    build_detector,
    parse_analyze_arg,
    parse_show_points,
)


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

            _annotate_motions(annotated_frame, keypoints_list, detector, analyze_motions, frame_count, detector.entity_label)
            if show_points:
                _draw_show_points(annotated_frame, keypoints_list, show_points, width, height, frame_count, detector.entity_label, depth_frame if ok_depth else None)
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
                _annotate_motions(annotated_frame, keypoints_list, detector, analyze_motions, frame_count, detector.entity_label)
                if show_points:
                    _draw_show_points(annotated_frame, keypoints_list, show_points, width, height, frame_count, detector.entity_label, depth_frame)
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
                _annotate_motions(annotated_frame, keypoints_list, detector, analyze_motions, frame_count, detector.entity_label)
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
        description="Pose detection using Orbbec camera or recorded files",
        parents=[build_base_parser()],
    )
    parser.add_argument("--source", type=str, default="camera",
                        help="Video file path or 'camera' for live Orbbec camera (default: camera)")
    parser.add_argument("--stream-type", type=str, default="color", choices=["color", "depth"],
                        help="Stream type: 'color' runs pose detection, 'depth' shows colorized depth")
    parser.add_argument("--depth-source", type=str, default=None,
                        help="Optional depth video file for side-by-side display")
    parser.add_argument("--camera-index", type=int, default=0, help="Orbbec camera index")

    args = parser.parse_args()
    detector = build_detector(args)

    if args.source == "camera":
        process_camera_stream(
            detector=detector,
            camera_index=args.camera_index,
            output_path=args.output,
            show=not args.no_show,
            analyze_motions=parse_analyze_arg(args.analyze),
            show_points=parse_show_points(args.show_points),
        )
    else:
        process_file_stream(
            detector=detector,
            source=args.source,
            stream_type=args.stream_type,
            depth_source=args.depth_source,
            output_path=args.output,
            show=not args.no_show,
            analyze_motions=parse_analyze_arg(args.analyze),
            show_points=parse_show_points(args.show_points),
        )


if __name__ == "__main__":
    main()
