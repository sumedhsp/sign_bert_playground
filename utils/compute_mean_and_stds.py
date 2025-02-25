import numpy as np
import os
import json
import argparse

def read_json(fpath):
    with open(fpath, 'r') as fid:
        data = json.load(fid)
    return data


def compute_means_stds(dataset,  skeleton_dpath):
    """Compute mean and standard deviation for the dataset."""
    all_data = []
    for sample in dataset:
        skeleton_video_fpath = os.path.join(skeleton_dpath, f"{sample}.npy")
        skeleton_data = np.load(skeleton_video_fpath)[dataset[sample]['action'][1]:dataset[sample]['action'][2]]
        skeleton_data = skeleton_data[..., :2]  # Keep only x, y coordinates
        all_data.append(skeleton_data)

    all_data = np.concatenate(all_data, axis=0)  # Stack all frames
    means = np.mean(all_data, axis=0)
    stds = np.std(all_data, axis=0)

    return means, stds

# Example usage with training dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=str)
    parser.add_argument("--skeleton", required=True, type=str)
    args = parser.parse_args()

    video_info = read_json(args.metadata)
    means, stds = compute_means_stds(video_info, args.skeleton)

    # Save to .npy files
    np.save("means.npy", means)
    np.save("stds.npy", stds)
