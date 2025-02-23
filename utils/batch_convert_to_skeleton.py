import os
import cv2
import numpy as np
import torch
import torchvision
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples

# Load Pose Estimation Model (RTMPose)
print (os.getcwd())
POSE_CONFIG = "sign_bert_playground/models/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py"
POSE_MODEL = "sign_bert_playground/models/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth"

original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load  # Override torch.load

device = "cuda" if torch.cuda.is_available() else "cpu"
pose_model = init_model(POSE_CONFIG, POSE_MODEL, device=device)

# Load Faster R-CNN from PyTorch
det_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
det_model.eval()

def detect_people(frame):
    """Detect people using PyTorch's Faster R-CNN."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = det_model(frame_tensor)[0]

    person_detections = []
    for i in range(len(detections["boxes"])):
        if detections["labels"][i] == 1 and detections["scores"][i] > 0.5:
            person_detections.append(detections["boxes"][i].cpu().numpy())

    return person_detections

def estimate_pose(frame, person_detections):
    """Estimate pose using RTMPose in PyTorch and return keypoints in (x, y, confidence) format."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)

    pose_results = inference_topdown(pose_model, frame_rgb, person_detections)
    pose_data = merge_data_samples(pose_results)
    
    # Check if pose_data contains valid keypoints
    if hasattr(pose_data, "pred_instances") and pose_data.pred_instances.keypoints is not None:
        keypoints = pose_data.pred_instances.keypoints  # Extract keypoints
        
        # Extract confidence scores from keypoint_scores
        if hasattr(pose_data.pred_instances, "keypoint_scores"):
            confidence = pose_data.pred_instances.keypoint_scores
        else:
            confidence = np.ones((133,))  # Default confidence=1 if not available

        # Ensure keypoints have shape (133, 2) and confidence has shape (133,)
        keypoints = keypoints.squeeze()  # Remove batch dimension if present
        confidence = confidence.squeeze()  # Remove batch dimension if present

        # Concatenate confidence scores to keypoints (x, y, confidence)
        keypoints_with_conf = np.hstack((keypoints, confidence.reshape(-1, 1)))  # Shape: (133, 3)

        return keypoints_with_conf

    return np.zeros((133, 3))  # Return empty keypoints if detection fails

def process_videos_in_batches(video_folder, output_folder, batch_size=5):
    """Process videos from a folder in batches and convert them to skeleton data."""
    os.makedirs(output_folder, exist_ok=True)

    # Get list of all video files in the folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith((".mp4", ".avi", ".mov"))]
    
    total_videos = len(video_files)
    print(f"Found {total_videos} videos. Processing in batches of {batch_size}.")

    for i in range(0, total_videos, batch_size):
        batch_videos = video_files[i:i + batch_size]  # Select batch of videos

        for video_file in batch_videos:
            video_path = os.path.join(video_folder, video_file)
            output_skeleton_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.npy")

            print(f"Processing {video_file}...")

            cap = cv2.VideoCapture(video_path)
            all_keypoints = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Step 1: Detect humans using PyTorch Faster R-CNN
                person_detections = detect_people(frame)

                if len(person_detections) == 0:
                    all_keypoints.append(np.zeros((133, 3)))  # 133 keypoints (x, y, confidence)
                    continue

                # Step 2: Estimate pose using RTMPose (PyTorch)
                keypoints = estimate_pose(frame, person_detections)
                all_keypoints.append(keypoints)

            cap.release()

            # Convert list to numpy array and save
            all_keypoints = np.array(all_keypoints)  # Shape: (num_frames, 133, 3)
            np.save(output_skeleton_path, all_keypoints)

            print(f"Saved skeleton data for {video_file} to {output_skeleton_path}")

    print("Batch processing completed.")

# Example usage
video_folder = "sign_bert_playground/WLASL2000/"
output_folder = "sign_bert_playground/wlasl_skeleton/"
process_videos_in_batches(video_folder, output_folder, batch_size=5)
