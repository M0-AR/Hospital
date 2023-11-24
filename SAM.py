import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import traceback

# Initialize and load the SAM model
CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

# Initialize the SamAutomaticMaskGenerator
sam_model = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100
)

# Function to process images and generate masks
def process_image_with_sam(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to load image from {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = sam_model.generate(image_rgb)
    return image_rgb, masks

# Function to show masks on the image
def show_anns(masks, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_autoscale_on(False)

    for mask_dict in masks:
        segmentation = mask_dict['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img = np.ones((segmentation.shape[0], segmentation.shape[1], 4))
        img[:, :, 3] = 0
        img[segmentation] = color_mask
        ax.imshow(img)

# Example usage
image_path = 'test.jpeg'  # Replace with your image path
try:
    image_rgb, masks = process_image_with_sam(image_path)

    # Display the image with masks
    plt.figure(figsize=(20,20))
    plt.imshow(image_rgb)
    show_anns(masks)
    plt.axis('off')
    plt.show()

except FileNotFoundError:
    traceback.print_exc()  # This will print the stack trace of the error
    print(f"File not found: {image_path}")
except Exception as e:
    traceback.print_exc()  # This will print the stack trace of the error
    print(f"An error occurred: {e}")