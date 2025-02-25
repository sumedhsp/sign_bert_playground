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
        
        # print (keypoints.shape)
        # Handle multiple detections (e.g., shape (2, 133, 2))
        if len(keypoints.shape) == 3 and keypoints.shape[0] > 1:
            keypoints = keypoints[0]  # Select first detected person
        
        print (keypoints.shape, keypoints.shape[0])
        # Ensure the shape is (133, 2) before proceeding
        if keypoints.shape[0] == 1:  # If batch dimension exists
            keypoints = keypoints.squeeze(0)  # Remove first dimension

        # Extract confidence scores from keypoint_scores
        if hasattr(pose_data.pred_instances, "keypoint_scores"):
            confidence = pose_data.pred_instances.keypoint_scores

            print (confidence.shape, "confidence")
            # Ensure confidence has the correct shape
            if confidence.shape[0] > 133:
                confidence = confidence[:133]  # Select first 133 scores

        else:
            confidence = np.ones((133,))  # Default confidence=1 if not available

        # Ensure keypoints have shape (133, 2) and confidence has shape (133,)
        keypoints = keypoints.squeeze()  # Remove batch dimension if present
        confidence = confidence.squeeze()  # Remove batch dimension if present

        # Concatenate confidence scores to keypoints (x, y, confidence)
        keypoints_with_conf = np.hstack((keypoints, confidence.reshape(-1, 1)))  # Shape: (133, 3)

        return keypoints_with_conf

    return np.zeros((133, 3))  # Return empty keypoints if detection fails

def read_file_to_list(file_path):
    """
    Opens a file, reads each line, and stores it in a list.

    Args:
        file_path (str): The path to the file.

    Returns:
        list: A list containing each line of the file.
              Returns an empty list if the file does not exist or an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            lines = [line.rstrip('\n') for line in file]
        return lines
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



def process_videos_in_batches(video_folder, output_folder, batch_size=5):
    """Process videos from a folder in batches and convert them to skeleton data."""
    os.makedirs(output_folder, exist_ok=True)
    print (os.getcwd())
    mismatched_videos_file = os.path.join(os.getcwd(), "mismatched_videos.txt")

    mismatched_vids = read_file_to_list(mismatched_videos_file)

    # Get list of all video files in the folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith((".mp4", ".avi", ".mov"))]
    
    # Get already processed videos
    processed_videos = {os.path.splitext(f)[0] for f in os.listdir(output_folder) if f.endswith(".npy")}

    total_videos = len(video_files)
    print(f"Found {total_videos} videos. Processing in batches of {batch_size}.")
    
    # Remove already processed videos from the list
    video_files = [f for f in video_files if os.path.splitext(f)[0] not in processed_videos]

    video_files = [f for f in video_files if f in mismatched_vids]
    print(f"Skipping {total_videos - len(video_files)} already processed videos.")

    mismatched_videos = []

    for i in range(0, total_videos, batch_size):
        batch_videos = video_files[i:i + batch_size]  # Select batch of videos

        for video_file in batch_videos:
            video_path = os.path.join(video_folder, video_file)
            output_skeleton_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.npy")

            print(f"Processing {video_file}...")


            cap = cv2.VideoCapture(video_path)
            all_keypoints = []

            mismatch_flag = False

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
                #try:
                keypoints = estimate_pose(frame, person_detections)
                all_keypoints.append(keypoints)
                #except Exception as e:
                #    print (f"Exception occured with {video_file}. Skipping for now!")
                #    mismatched_videos.append(video_file)
                #    mismatch_flag = True
                #    break

            cap.release()

            if (mismatch_flag):
                continue

            # Convert list to numpy array and save
            all_keypoints = np.array(all_keypoints)  # Shape: (num_frames, 133, 3)
            np.save(output_skeleton_path, all_keypoints)

            print(f"Saved skeleton data for {video_file} to {output_skeleton_path}")


    # Log mismatched videos
    if mismatched_videos:
        with open(mismatched_videos_file, "w") as f:
            for video in mismatched_videos:
                f.write(video + "\n")
        print(f"Logged {len(mismatched_videos)} mismatched videos in {mismatched_videos_file}")

    print("Batch processing completed.")

# Example usage
video_folder = "sign_bert_playground/WLASL2000/"
output_folder = "sign_bert_playground/wlasl_skeleton/"
process_videos_in_batches(video_folder, output_folder, batch_size=5)
