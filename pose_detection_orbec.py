import argparse
import time
from typing import List, Optional

import cv2
import numpy as np

from pose_detection import CleanPoseDetector, DEBUG


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


def process_orbbec_stream(
    detector: CleanPoseDetector,
    camera_index: int = 0,
    output_path: Optional[str] = None,
    show: bool = True,
    analyze_motions: Optional[List[str]] = None,
) -> None:
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

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
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

            y_offset = 30
            for person_id, keypoints in enumerate(keypoints_list):
                if analyze_motions:
                    for motion_type in analyze_motions:
                        action, normalized, angle = detector.analyze_motion(keypoints, motion_type)
                        if DEBUG:
                            if angle is not None:
                                print(
                                    f"Frame {frame_count} - Person {person_id + 1} - "
                                    f"{action}: {normalized:.2f} (angle: {angle:.0f}deg)"
                                )
                            else:
                                print(f"Frame {frame_count} - Person {person_id + 1} - {action}: N/A")
                        if angle is not None:
                            text = f"P{person_id + 1} {action}: {normalized:.2f}"
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

            info_text = f"Frame: {frame_count} | Persons: {len(keypoints_list)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pose detection using Orbbec camera color + depth streams"
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Orbbec camera index")
    parser.add_argument("--model", type=str, default="yolo26m-pose.pt", help="YOLO model name")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Output path for annotated color video")
    parser.add_argument("--no-show", action="store_true", help="Do not display preview windows")
    parser.add_argument(
        "--analyze",
        type=str,
        default="",
        help="Motions to analyze, e.g., {right_arm_abduction,left_arm_abduction}",
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

    process_orbbec_stream(
        detector=detector,
        camera_index=args.camera_index,
        output_path=args.output,
        show=not args.no_show,
        analyze_motions=parse_analyze_arg(args.analyze),
    )


if __name__ == "__main__":
    main()
