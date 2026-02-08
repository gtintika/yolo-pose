# yolo-pose-detection
# Copyright (C) <2026>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#
# Pose Detection with Duplicate Removal
# Filters out duplicate person detections (same person detected multiple times)
#

import cv2
import argparse
import time
from ultralytics import YOLO
import numpy as np
import math
from typing import Tuple, Optional, List, Dict

DEBUG = False  # Set to True to enable detailed debug prints
KEYPOINT_CONF_THRESHOLD = 0.5  # Minimum confidence for keypoint validity

class MotionAnalyzer:
    """Analyzes pose keypoints to detect motions and actions"""
    
    @staticmethod
    def calculate_angle_3points(point1: Dict, point2: Dict, point3: Dict) -> Optional[float]:
        """
        Calculate angle between three points (in degrees)
        
        Args:
            point1, point2, point3: Dictionaries with 'x', 'y', 'confidence' keys
            
        Returns:
            Angle in degrees (0-180) or None if points invalid
        """
        # Check confidence
        if any(p['confidence'] < KEYPOINT_CONF_THRESHOLD for p in [point1, point2, point3]):
            return None
        
        # Vector from point2 to point1
        v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
        # Vector from point2 to point3
        v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
        
        # Calculate angle
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        if mag1 == 0 or mag2 == 0:
            return None
        
        cos_angle = np.dot(v1, v2) / (mag1 * mag2)
        angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
        
        return angle
    
    @staticmethod
    def calculate_vertical_angle(shoulder: Dict, wrist: Dict, hip: Dict) -> Optional[float]:
        """
        Calculate angle of arm with vertical body axis
        
        Args:
            shoulder: Shoulder keypoint
            wrist: Wrist keypoint  
            hip: Hip keypoint (to determine vertical axis)
            
        Returns:
            Angle in degrees (0-180) or None if points invalid
            0° = arm down (next to body)
            90° = arm horizontal
            180° = arm up (parallel to body)
        """
        # Check confidence
        if any(p['confidence'] < KEYPOINT_CONF_THRESHOLD for p in [shoulder, wrist, hip]):
            return None
        
        # Downward vertical vector (pointing from shoulder to hip - direction of arm when down)
        # In image coords: Y increases downward, so this is the reference for "arm down"
        vertical_down = np.array([hip['x'] - shoulder['x'], hip['y'] - shoulder['y']])
        
        # Arm vector (from shoulder to wrist)
        arm = np.array([wrist['x'] - shoulder['x'], wrist['y'] - shoulder['y']])
        
        # Calculate angle
        mag_v = np.linalg.norm(vertical_down)
        mag_a = np.linalg.norm(arm)
        
        if mag_v == 0 or mag_a == 0:
            return None
        
        cos_angle = np.dot(vertical_down, arm) / (mag_v * mag_a)
        angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
        
        # This gives us:
        # - angle ≈ 0° when arm points down (parallel to vertical_down)
        # - angle ≈ 90° when arm is horizontal
        # - angle ≈ 180° when arm points up (opposite to vertical_down)
        
        return angle
    
    @staticmethod
    def analyze_arm_abduction(keypoints: List[Dict], side: str = 'right') -> Tuple[str, float, Optional[float]]:
        """
        Analyze arm abduction (raising arm to the side)
        
        Args:
            keypoints: List of keypoint dictionaries for one person
            side: 'right' or 'left'
            
        Returns:
            Tuple of (action_name, normalized_value, angle_degrees)
            - action_name: e.g., "right arm - abduction"
            - normalized_value: 0.0 (arm down) to 1.0 (arm up/parallel to body)
            - angle_degrees: actual angle in degrees (0-180)
        """
        # Get required keypoints
        shoulder_name = f'{side}_shoulder'
        elbow_name = f'{side}_elbow'
        wrist_name = f'{side}_wrist'
        hip_name = f'{side}_hip'

        shoulder = next((kpt for kpt in keypoints if kpt['name'] == shoulder_name), None)
        elbow = next((kpt for kpt in keypoints if kpt['name'] == elbow_name), None)
        wrist = next((kpt for kpt in keypoints if kpt['name'] == wrist_name), None)
        hip = next((kpt for kpt in keypoints if kpt['name'] == hip_name), None)

        if not all([shoulder, wrist, hip]):
            return (f"{side} arm - abduction", 0.0, None)

        # Check arm is extending laterally (not crossing body)
        # Right arm: elbow & wrist should be to the right of shoulder (greater x)
        # Left arm: elbow & wrist should be to the left of shoulder (lesser x)
        if elbow and elbow['confidence'] >= KEYPOINT_CONF_THRESHOLD:
            if side == 'right' and (elbow['x'] < shoulder['x'] or wrist['x'] < shoulder['x']):
                return (f"{side} arm - abduction", 0.0, None)
            if side == 'left' and (elbow['x'] > shoulder['x'] or wrist['x'] > shoulder['x']):
                return (f"{side} arm - abduction", 0.0, None)

        # Calculate angle with vertical body axis
        angle = MotionAnalyzer.calculate_vertical_angle(shoulder, wrist, hip)

        if angle is None:
            return (f"{side} arm - abduction", 0.0, None)
        
        # Normalize: 0° → 0.0, 180° → 1.0
        # few people will have perfect 180° abduction, 
        # so we can cap the normalized value at a practical maximum
        normalized = angle / 180.0

        # Explicit practical maximum threshold
        if normalized > 170.0 / 180.0:  # ~0.944
            normalized = 1.0
        
        return (f"{side} arm - abduction", normalized, angle)
    
    @staticmethod
    def analyze_elbow_flexion(keypoints: List[Dict], side: str = 'right') -> Tuple[str, float, Optional[float]]:
        """
        Analyze elbow flexion (bending elbow)
        
        Returns:
            Tuple of (action_name, normalized_value, angle_degrees)
            - normalized_value: 0.0 (0° angle) to 1.0 (180° angle)
            - 0.0 = very bent/acute angle
            - 1.0 = straight arm
        """
        shoulder_name = f'{side}_shoulder'
        elbow_name = f'{side}_elbow'
        wrist_name = f'{side}_wrist'
        
        shoulder = next((kpt for kpt in keypoints if kpt['name'] == shoulder_name), None)
        elbow = next((kpt for kpt in keypoints if kpt['name'] == elbow_name), None)
        wrist = next((kpt for kpt in keypoints if kpt['name'] == wrist_name), None)
        
        if not all([shoulder, elbow, wrist]):
            return (f"{side} elbow - flexion", 0.0, None)
        
        angle = MotionAnalyzer.calculate_angle_3points(shoulder, elbow, wrist)
        
        if angle is None:
            return (f"{side} elbow - flexion", 0.0, None)
        
        # Normalize: 0° → 0.0, 180° → 1.0 (direct angle mapping)
        normalized = angle / 180.0
        
        return (f"{side} elbow - flexion", normalized, angle)
    
    @staticmethod
    def analyze_knee_flexion(keypoints: List[Dict], side: str = 'right') -> Tuple[str, float, Optional[float]]:
        """
        Analyze knee flexion (bending knee)
        
        Returns:
            Tuple of (action_name, normalized_value, angle_degrees)
            - normalized_value: 0.0 (0° angle) to 1.0 (180° angle)
            - 0.0 = very bent/acute angle
            - 1.0 = straight leg
        """
        hip_name = f'{side}_hip'
        knee_name = f'{side}_knee'
        ankle_name = f'{side}_ankle'
        
        hip = next((kpt for kpt in keypoints if kpt['name'] == hip_name), None)
        knee = next((kpt for kpt in keypoints if kpt['name'] == knee_name), None)
        ankle = next((kpt for kpt in keypoints if kpt['name'] == ankle_name), None)
        
        if not all([hip, knee, ankle]):
            return (f"{side} knee - flexion", 0.0, None)
        
        angle = MotionAnalyzer.calculate_angle_3points(hip, knee, ankle)
        
        if angle is None:
            return (f"{side} knee - flexion", 0.0, None)
        
        # Normalize: 0° → 0.0, 180° → 1.0 (direct angle mapping)
        normalized = angle / 180.0
        
        return (f"{side} knee - flexion", normalized, angle)


class PoseDetector:
    def __init__(self, model_name='yolo26m-pose.pt', conf_threshold=0.5, device=None):
        """
        Initialize the pose detector

        Args:
            model_name: Name of the YOLO pose model (default: yolo26m-pose.pt)
            conf_threshold: Confidence threshold for detections
            device: Device for inference (cpu, mps, cuda, cuda:0, etc.)
        """
        print(f"Loading model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.device = device
        self.motion_analyzer = MotionAnalyzer()
        
        # COCO keypoint indices (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def detect_pose(self, frame: np.ndarray, normalize_coords: bool = True) -> Tuple[np.ndarray, List[List[Dict]]]:
        """
        Unified pose detection function for any media (image or video frame)
        
        Args:
            frame: Input frame as numpy array (from image or video)
            normalize_coords: If True, returns normalized coordinates (0-1)
            
        Returns:
            Tuple of (annotated_frame, keypoints_list)
            - annotated_frame: Frame with pose visualization
            - keypoints_list: List of keypoints for each detected person
        """
        # Run inference
        kwargs = dict(conf=self.conf_threshold, verbose=False)
        if self.device:
            kwargs['device'] = self.device
        results = self.model(frame, **kwargs)

        # Annotate frame
        annotated_frame = results[0].plot()

        # Extract keypoints
        keypoints_list = self._extract_keypoints(results[0], normalize_coords)

        return annotated_frame, keypoints_list
    
    def _extract_keypoints(self, results, normalized: bool = True) -> List[List[Dict]]:
        """
        Extract keypoints from YOLO results
        
        Args:
            results: YOLO results object
            normalized: If True, returns normalized coordinates (xyn)
            
        Returns:
            List of keypoints for each detected person
        """
        keypoints_data = []
        
        if results.keypoints is not None:
            # Choose between normalized (xyn) or pixel (xy) coordinates
            keypoints_tensor = results.keypoints.xyn if normalized else results.keypoints.data
            
            for person_idx, person_kpts in enumerate(keypoints_tensor):
                person_data = []
                for i, kpt in enumerate(person_kpts):
                    if normalized:
                        # xyn format: only x, y (normalized 0-1)
                        x, y = kpt
                        # Get confidence from separate tensor
                        conf = float(results.keypoints.conf[person_idx][i]) if results.keypoints.conf is not None else 0.0
                    else:
                        # xy format: x, y (pixels), confidence
                        x, y, conf = kpt
                    
                    person_data.append({
                        'name': self.keypoint_names[i],
                        'x': float(x),
                        'y': float(y),
                        'confidence': float(conf)
                    })
                keypoints_data.append(person_data)
        
        return keypoints_data
    
    def analyze_motion(self, keypoints: List[Dict], motion_type: str = 'right_arm_abduction') -> Tuple[str, float, Optional[float]]:
        """
        Analyze motion/action from keypoints
        
        Args:
            keypoints: List of keypoint dictionaries for one person
            motion_type: Type of motion to analyze. Options:
                - 'right_arm_abduction', 'left_arm_abduction'
                - 'right_elbow_flexion', 'left_elbow_flexion'
                - 'right_knee_flexion', 'left_knee_flexion'
        
        Returns:
            Tuple of (action_name, normalized_value, optional_angle)
            Example: ("right arm - abduction", 0.5, 90.0)
        """
        motion_map = {
            'right_arm_abduction': lambda: self.motion_analyzer.analyze_arm_abduction(keypoints, 'right'),
            'left_arm_abduction': lambda: self.motion_analyzer.analyze_arm_abduction(keypoints, 'left'),
            'right_elbow_flexion': lambda: self.motion_analyzer.analyze_elbow_flexion(keypoints, 'right'),
            'left_elbow_flexion': lambda: self.motion_analyzer.analyze_elbow_flexion(keypoints, 'left'),
            'right_knee_flexion': lambda: self.motion_analyzer.analyze_knee_flexion(keypoints, 'right'),
            'left_knee_flexion': lambda: self.motion_analyzer.analyze_knee_flexion(keypoints, 'left'),
        }
        
        if motion_type in motion_map:
            return motion_map[motion_type]()
        else:
            return ("unknown", 0.0, None)
    
    def process_media(self, source, output_path: Optional[str] = None, 
                     show: bool = True, analyze_motions: List[str] = None) -> None:
        """
        Unified function to process any media (image, video, or camera)
        
        Args:
            source: Image path, video path, or camera index (int)
            output_path: Optional path to save output
            show: Whether to display results
            analyze_motions: List of motions to analyze (e.g., ['right_arm_abduction'])
        """
        # Determine if source is image or video/camera
        is_image = isinstance(source, str) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        
        if is_image:
            self._process_image(source, output_path, show, analyze_motions)
        else:
            self._process_video(source, output_path, show, analyze_motions)
    
    def _process_image(self, image_path: str, output_path: Optional[str], 
                      show: bool, analyze_motions: Optional[List[str]]) -> None:
        """Process single image"""
        print(f"Processing image: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image from {image_path}")
            return
        
        # Detect pose
        annotated_frame, keypoints_list = self.detect_pose(frame, normalize_coords=True)
        
        print(f"Detected {len(keypoints_list)} person(s)")
        
        # Analyze motions if requested
        if analyze_motions:
            for person_id, keypoints in enumerate(keypoints_list):
                print(f"\nPerson {person_id + 1}:")
                for motion_type in analyze_motions:
                    action, normalized, angle = self.analyze_motion(keypoints, motion_type)
                    if angle is not None:
                        print(f"  {action}: {normalized:.3f} (angle: {angle:.1f}°)")
                    else:
                        print(f"  {action}: N/A (insufficient confidence)")
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            print(f"\nSaved result to: {output_path}")
        
        # Display
        if show:
            cv2.imshow('Pose Detection', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def _process_video(self, video_source, output_path: Optional[str], 
                      show: bool, analyze_motions: Optional[List[str]]) -> None:
        """Process video stream (camera or video file)"""
        # Open video source
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"Video: {width}x{height} @ {fps}fps")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Recording to: {output_path}")
        
        print("Processing video... Press 'q' to quit")

        frame_count = 0
        inference_times = []
        processing_times = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break

                frame_start = time.perf_counter()

                # Detect pose
                annotated_frame, keypoints_list = self.detect_pose(frame, normalize_coords=True)

                inference_time = time.perf_counter() - frame_start
                inference_times.append(inference_time)
                
                # Analyze motions and overlay on frame
                y_offset = 30
                for person_id, keypoints in enumerate(keypoints_list):
                    if analyze_motions:
                        for motion_type in analyze_motions:
                            action, normalized, angle = self.analyze_motion(keypoints, motion_type)
                            if DEBUG == True:                            
                                print(f"Frame {frame_count} - Person {person_id + 1} - {action}: {normalized:.2f} (angle: {angle:.0f}°)" if angle is not None else f"Frame {frame_count} - Person {person_id + 1} - {action}: N/A") 
                            if angle is not None:
                                text = f"P{person_id+1} {action}: {normalized:.2f}"
                                cv2.putText(annotated_frame, text, (10, y_offset),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                y_offset += 25
                
                # Add frame info
                info_text = f"Frame: {frame_count} | Persons: {len(keypoints_list)}"
                cv2.putText(annotated_frame, info_text, (10, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display
                if show:
                    cv2.imshow('Pose Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quitting...")
                        break
                
                processing_time = time.perf_counter() - frame_start
                processing_times.append(processing_time)

                frame_count += 1

        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

            # Print stats
            if frame_count > 0:
                avg_inference = sum(inference_times) / len(inference_times) * 1000
                avg_processing = sum(processing_times) / len(processing_times) * 1000
                total_time = sum(processing_times)
                print(f"\n--- Stats ---")
                print(f"Frames processed: {frame_count}")
                print(f"Avg inference:    {avg_inference:.1f} ms/frame")
                print(f"Avg processing:   {avg_processing:.1f} ms/frame (inference + annotation + I/O)")
                print(f"Throughput:       {frame_count / total_time:.1f} FPS")
                print(f"Total time:       {total_time:.2f} s")

class DuplicateDetectionFilter:
    """Removes duplicate person detections based on keypoint similarity"""
    
    def __init__(self, iou_threshold: float = 0.5, keypoint_similarity_threshold: float = 0.7):
        """
        Initialize duplicate filter
        
        Args:
            iou_threshold: IoU threshold for bounding box overlap (0-1)
            keypoint_similarity_threshold: Similarity threshold for keypoints (0-1)
                Lower = more aggressive filtering. 0.7 catches most duplicates.
        """
        self.iou_threshold = iou_threshold
        self.keypoint_similarity_threshold = keypoint_similarity_threshold
    
    @staticmethod
    def calculate_keypoint_similarity(kpts1: List[Dict], kpts2: List[Dict]) -> float:
        """
        Calculate similarity between two sets of keypoints
        
        Returns:
            Similarity score (0-1), where 1 = identical keypoints
        """
        if len(kpts1) != len(kpts2):
            return 0.0
        
        total_distance = 0.0
        valid_pairs = 0
        
        for kpt1, kpt2 in zip(kpts1, kpts2):
            # Only compare keypoints with sufficient confidence
            if kpt1['confidence'] > 0.3 and kpt2['confidence'] > 0.3:
                # Calculate Euclidean distance (normalized coordinates)
                distance = np.sqrt(
                    (kpt1['x'] - kpt2['x'])**2 + 
                    (kpt1['y'] - kpt2['y'])**2
                )
                total_distance += distance
                valid_pairs += 1
        
        if valid_pairs == 0:
            return 0.0
        
        # Average distance
        avg_distance = total_distance / valid_pairs
        
        # Convert to similarity (closer = more similar)
        # Distance of 0.1 (10% of image) or less = very similar
        # Using 0.2 divisor so detections up to 20% apart still get partial similarity
        similarity = max(0.0, 1.0 - (avg_distance / 0.2))
        
        return similarity
    
    @staticmethod
    def calculate_bbox_iou(box1, box2) -> float:
        """
        Calculate Intersection over Union of two bounding boxes
        
        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            IoU score (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def remove_duplicates(self, keypoints_list: List[List[Dict]], 
                         results) -> List[List[Dict]]:
        """
        Remove duplicate detections of the same person
        
        Args:
            keypoints_list: List of keypoints for all detected persons
            results: YOLO results object
            
        Returns:
            Filtered list with duplicates removed
        """
        if len(keypoints_list) <= 1:
            return keypoints_list
        
        # Get bounding boxes if available
        bboxes = []
        if results.boxes is not None:
            for box in results.boxes:
                bboxes.append(box.xyxy[0].cpu().numpy())
        
        # Track which detections to keep
        keep_indices = []
        used_indices = set()
        
        for i in range(len(keypoints_list)):
            if i in used_indices:
                continue
            
            # This detection is unique so far
            keep_indices.append(i)
            
            # Check all remaining detections for duplicates
            for j in range(i + 1, len(keypoints_list)):
                if j in used_indices:
                    continue
                
                # Calculate keypoint similarity
                similarity = self.calculate_keypoint_similarity(
                    keypoints_list[i], 
                    keypoints_list[j]
                )
                
                # Calculate bbox IoU if available
                iou = 0.0
                if i < len(bboxes) and j < len(bboxes):
                    iou = self.calculate_bbox_iou(bboxes[i], bboxes[j])
                
                # Mark as duplicate if very similar
                is_duplicate = (
                    similarity > self.keypoint_similarity_threshold or
                    iou > self.iou_threshold
                )
                
                if is_duplicate:
                    used_indices.add(j)
                    # Keep the one with higher average confidence
                    conf_i = np.mean([kpt['confidence'] for kpt in keypoints_list[keep_indices[-1]]])
                    conf_j = np.mean([kpt['confidence'] for kpt in keypoints_list[j]])
                    
                    if conf_j > conf_i:
                        # Replace current kept index with j (better confidence)
                        used_indices.add(keep_indices[-1])
                        used_indices.remove(j)
                        keep_indices[-1] = j
        
        # Return filtered list
        return [keypoints_list[i] for i in keep_indices]


class CleanPoseDetector(PoseDetector):
    """Pose detector with duplicate removal"""
    
    def __init__(self, model_name='yolo26m-pose.pt', conf_threshold=0.5,
                 remove_duplicates=True, duplicate_threshold=0.7, device=None):
        """
        Initialize clean detector

        Args:
            model_name: YOLO model name
            conf_threshold: Detection confidence threshold
            remove_duplicates: Whether to remove duplicate detections
            duplicate_threshold: Keypoint similarity threshold for duplicates
            device: Device for inference (cpu, mps, cuda, cuda:0, etc.)
        """
        super().__init__(model_name, conf_threshold, device=device)
        self.remove_duplicates = remove_duplicates
        self.duplicate_filter = DuplicateDetectionFilter(
            keypoint_similarity_threshold=duplicate_threshold
        )
    
    def detect_pose(self, frame: np.ndarray, normalize_coords: bool = True) -> Tuple[np.ndarray, List[List[Dict]]]:
        """
        Detect pose with duplicate removal
        
        Args:
            frame: Input frame
            normalize_coords: Return normalized coordinates
            
        Returns:
            (annotated_frame, keypoints_list)
            Keypoints list has duplicates removed if enabled
        """
        # Run inference
        kwargs = dict(conf=self.conf_threshold, verbose=False)
        if self.device:
            kwargs['device'] = self.device
        results = self.model(frame, **kwargs)

        # Annotate frame
        annotated_frame = results[0].plot()

        # Extract all keypoints
        keypoints_list = self._extract_keypoints(results[0], normalize_coords)
        
        # Remove duplicates if enabled
        if self.remove_duplicates and len(keypoints_list) > 1:
            original_count = len(keypoints_list)
            keypoints_list = self.duplicate_filter.remove_duplicates(
                keypoints_list, 
                results[0]
            )
            
            if len(keypoints_list) < original_count:
                # Optionally log duplicate removal
                pass
        
        return annotated_frame, keypoints_list


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pose Detection with Duplicate Removal')
    parser.add_argument('--source', type=str, required=True,
                       help='Video file path or camera index')
    parser.add_argument('--model', type=str, default='yolo26m-pose.pt',
                       help='YOLO model name')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display video')
    parser.add_argument('--analyze', type=str, default='',
                       help='Motions to analyze, e.g., {right_arm_abduction,left_arm_abduction}')
    parser.add_argument('--device', type=str, default=None,
                       help='Device for inference: cpu, mps, cuda, or cuda:0 (default: auto)')
    parser.add_argument('--keep-duplicates', action='store_true',
                       help='Keep duplicate detections (disable filtering)')
    parser.add_argument('--duplicate-threshold', type=float, default=0.7,
                       help='Keypoint similarity threshold for duplicates (0-1, lower = more aggressive)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = CleanPoseDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        remove_duplicates=not args.keep_duplicates,
        duplicate_threshold=args.duplicate_threshold,
        device=args.device
    )
    
    # Process video
    source = int(args.source) if args.source.isdigit() else args.source
    
    analyze_motions = None
    if args.analyze:
        cleaned = args.analyze.strip('{}')
        analyze_motions = [m.strip() for m in cleaned.split(',') if m.strip()]

    detector.process_media(
        source=source,
        output_path=args.output,
        show=not args.no_show,
        analyze_motions=analyze_motions
    )


if __name__ == '__main__':
    main()
