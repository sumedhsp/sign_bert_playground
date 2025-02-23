import os
import cv2
import numpy as np
import torch
import torchvision
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import mmcv

# Load Pose Estimation Model (RTMPose)
POSE_CONFIG = "../models/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py"
POSE_MODEL = "../models/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth"

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

def extract_skeleton(video_path, save_path, video_id):

    """Extract skeleton keypoints from a video using RTMPose & PyTorch's Faster R-CNN."""

    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video

        # Step 1: Detect humans using PyTorch Faster R-CNN
        #print ("Step 1: Detecting humans")
        person_detections = detect_people(frame)

        if len(person_detections) == 0:
            all_keypoints.append(np.zeros((133, 3)))  # 133 keypoints (x, y, confidence)
            continue

        # Step 2: Convert frame to tensor for RTMPose
        #print ("Step 2: Converting frame to tensor for RTMPose")
        frame_rgb = mmcv.imconvert(frame, 'bgr', 'rgb')  # Convert BGR to RGB
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)  
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 3: Estimate pose using RTMPose
        #print ("Step 3: Estimating pose using RTMPose")
        pose_results = inference_topdown(pose_model, frame_rgb, person_detections)
        pose_data = merge_data_samples(pose_results)

        if hasattr(pose_data, "pred_instances") and pose_data.pred_instances.keypoints is not None:
            keypoints = pose_data.pred_instances.keypoints  # Shape: (133, 3)
            confidence = pose_data.pred_instances.keypoint_scores

            keypoints = keypoints.squeeze()
            confidence = confidence.squeeze()

            keypoints_with_conf = np.hstack((keypoints, confidence.reshape(-1, 1)))

            all_keypoints.append(keypoints_with_conf)
        else:
            all_keypoints.append(np.zeros((133, 3)))  # Fill with zeros if no keypoints detected

    cap.release()
    
    # Convert list to numpy array and save
    print ("Saving the keypoints")
    all_keypoints = np.array(all_keypoints)  # Shape: (num_frames, 133, 3)
    np.save(save_path, all_keypoints)
    print(f"Saved {video_id} skeleton data to {save_path}")

# Example usage
video_file = "../WLASL2000/67823.mp4"
skeleton_save_path = "../wlasl_skeleton/67823.npy"
extract_skeleton(video_file, skeleton_save_path)


