#TODO: Update this.

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

test_path = "/data/datasets/sound_datasets/pytorch_UrbanSound8K/image_80tr10va10te/trainset/children_playing/60935-2-0-9.jpg"
img = Image.open(test_path)
plt.imshow(img)
plt.tight_layout()

img_arr = np.array(img)

plt.figure(figsize=(16, 8))

plt.subplot(1,3,1)
plt.imshow(img_arr[:,:,0],cmap='Reds')
plt.title("Red channel")
plt.subplot(1,3,2)
plt.imshow(img_arr[:,:,1],cmap='Greens')
plt.title("Green channel")
plt.subplot(1,3,3)
plt.imshow(img_arr[:,:,2],cmap='Blues')
plt.title("Blue channel")

plt.tight_layout()