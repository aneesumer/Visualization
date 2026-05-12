import numpy as np
import imageio.v3 as imageio
import matplotlib
import matplotlib.pyplot as plt

# Load image
img = imageio.imread("BlueAndYellowMacaw_AraArarauna.jpg")
img = img / 255.


def rotate_hue(image_rgb: np.ndarray, phi: float) -> np.ndarray:
    img_hsv = matplotlib.colors.rgb_to_hsv(image_rgb)
    shift = phi / (2 * np.pi)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + shift) % 1.0
    return matplotlib.colors.hsv_to_rgb(img_hsv)


phis = [k / 5 * 2 * np.pi for k in range(5)]
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for ax, phi, k in zip(axes, phis, range(5)):
    rotated = rotate_hue(img, phi)
    ax.imshow(rotated)
    ax.set_title(f"k={k}, φ = {k}/5 · 2π", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("exercise_1_3_hue_rotation.png", dpi=150, bbox_inches="tight")
plt.show()
