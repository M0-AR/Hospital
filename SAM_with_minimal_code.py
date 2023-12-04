# https://ybagoury.medium.com/metas-sam-easy-image-segmentation-with-minimal-code-ed591c4319e8
from transformers import pipeline

generator = pipeline("mask-generation", model="facebook/sam-vit-large", device=0)

from PIL import Image

# Local Image
local_image = Image.open("./6010_0.0_crop_0.6.png").convert("RGB")

import matplotlib.pyplot as plt

plt.imshow(local_image)


import numpy as np
import matplotlib.pyplot as plt
import gc

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()


def show_masks_on_image(raw_image, masks):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
      show_mask(mask, ax=ax, random_color=True)
  plt.axis("off")
  plt.show()
  del mask
  gc.collect()
  


outputs = generator(local_image, points_per_batch=64)

masks = outputs["masks"]
show_masks_on_image(local_image, masks)