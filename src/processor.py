# pylint: disable=no-member, unused-variable, no-name-in-module, too-many-locals

"""
Algorithm
"""

import random
from typing import Optional
import cv2
import numpy as np
from skimage import exposure
from skimage.filters import unsharp_mask
from matplotlib import pyplot as plt
from .utils import EvalMetrics, salt_pepper_noise

SEED = 108

random.seed(SEED)
np.random.seed(SEED)


class ImageProcessor:
    """
    Class to process an image and compute evaluation metrics.
    """

    def __init__(self):
        self.img = None
        self.img_gray = None
        self.processed_imgs = None
        self.metrics = None
        self.noise_config = None
        self.img_noisy = None

    def run(self, path: str, noise_type: Optional[str] = None, **kwargs) -> None:
        """
        Calls process and evaluate to process the image and calculate metrics.

        Args:
            path (str): path to the image
            noise_type (Optional[str], optional): Type of noise to inject (gaussian
                                                  or salt_pepper). Defaults to None.
        """
        self.process(path, noise_type, **kwargs)
        processed_imgs = [
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in self.processed_imgs
        ]
        self.evaluate(self.img_gray, processed_imgs)

    def add_noise(self, img: np.ndarray, noise_type: str, **kwargs) -> np.ndarray:
        """
        Injects noise into the given image.

        Args:
            img (np.ndarray): The input image to which noise should be added.
            noise_type (str): Type of noise to inject (gaussian or salt_pepper)

        Raises:
            ValueError: If noise_type isn't gaussian or salt_pepper.

        Returns:
            np.ndarray: Image injected with noise.
        """

        noise_type = noise_type.lower()  # Case-insensitive handling

        if noise_type == "salt_pepper":
            percentage = float(kwargs.get("percentage", 0.10))
            noise = salt_pepper_noise(img, percentage)
            noise = np.expand_dims(noise, axis=-1)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            self.noise_config = {"noise_type": noise_type, "percentage": percentage}
        elif noise_type == "gaussian":
            mean = float(kwargs.get("mean", 0))
            variance = float(kwargs.get("variance", 1))
            stddev = np.sqrt(variance)
            gaussian_noise = np.random.normal(mean, stddev, img.shape).astype(np.uint8)
            img = cv2.add(img, gaussian_noise)
            self.noise_config = {
                "noise_type": noise_type,
                "mean": mean,
                "variance": variance,
            }
        else:
            raise ValueError(
                f"Invalid noise_type: {noise_type}. Must be `gaussian` or `salt_pepper`"
            )

        return img

    def process(self, path: str, noise_type: Optional[str] = None, **kwargs) -> list:
        """
        Processes the images and forms various combinations.

        Args:
            path (str): path to the image
            noise_type (Optional[str], optional): Type of noise to inject (gaussian or salt_pepper).
                                                  Defaults to None.

        Returns:
            list: Contains 6 processed images.
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_noisy = self.add_noise(img, noise_type, **kwargs) if noise_type else img

        img_gray = cv2.cvtColor(img_noisy, cv2.COLOR_RGB2GRAY)

        # 1. Gaussian filter ---> CLAHE --->  Unsharp-Mask
        gaus_img = cv2.GaussianBlur(img_gray, (5, 5), 0, borderType=cv2.BORDER_CONSTANT)
        hist_equ = exposure.equalize_adapthist(gaus_img)
        unsharp_bilat = unsharp_mask(hist_equ, radius=7, amount=2)

        # 2. Split channels ---> Median Blur 2x ---> CLAHE --->  Gamma filter
        r, g, b = cv2.split(img_noisy)
        img_med = cv2.medianBlur(g, 3)
        img_med_1 = cv2.medianBlur(img_med, 3)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        cl_img = clahe.apply(img_med_1)
        gamma_img = np.array(255 * (cl_img / 255) ** 0.8, dtype="uint8")

        # Merging the 3 pre-processed imgs
        l, m = gamma_img.shape
        final_img = np.ones((l, m, 3))
        final_img[:, :, 0] = final_img[:, :, 0] * gamma_img
        final_img[:, :, 1] = final_img[:, :, 1] * unsharp_bilat
        final_img[:, :, 2] = final_img[:, :, 2] * g

        d1, d2, d3 = cv2.split(final_img)

        permutations = [
            cv2.merge((d1, d2, d3)).astype(np.uint8),
            cv2.merge((d1, d3, d2)).astype(np.uint8),
            cv2.merge((d2, d1, d3)).astype(np.uint8),
            cv2.merge((d2, d3, d1)).astype(np.uint8),
            cv2.merge((d3, d1, d2)).astype(np.uint8),
            cv2.merge((d3, d2, d1)).astype(np.uint8),
        ]

        self.img = img
        self.img_gray = img_gray
        self.img_noisy = img_noisy if noise_type else None
        self.processed_imgs = permutations

    def evaluate(self, img_gray: np.array, permutations: list) -> None:
        """
        Calculates various metrics for processed image combinations.

        Args:
            img_gray (np.array): Grayscale image to calculate metrics.
            permutations (list): The list of different combinations of images obtained
                                 after running process method.
        """
        psnr_list = [
            EvalMetrics.calculate_psnr(img_gray, perm) for perm in permutations
        ]

        correlation_list = [
            EvalMetrics.calculate_correlation(img_gray, perm) for perm in permutations
        ]

        epi_list = [
            EvalMetrics.edge_preservation_index(img_gray, perm) for perm in permutations
        ]

        ssim_list = [
            EvalMetrics.calculate_ssim(img_gray, perm) for perm in permutations
        ]

        self.metrics = {
            "psnr": psnr_list,
            "corr": correlation_list,
            "epi": epi_list,
            "ssim": ssim_list,
        }

    def plot(self) -> None:
        """
        Plots the original image and the images obtained
        after processing.
        """

        original = self.img
        permutations = self.processed_imgs
        fig, axes = plt.subplots(2, 4, figsize=(15, 5))

        axes[0, 0].imshow(original)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        if self.img_noisy is not None:
            axes[0, 1].imshow(self.img_noisy)
            axes[0, 1].set_title("Noisy Image")
            axes[0, 1].axis("off")
        else:
            axes[0, 1].imshow(self.img_gray, cmap="gray")
            axes[0, 1].set_title("Grayscale Image")
            axes[0, 1].axis("off")

        permutation_labels = [
            "123",
            "132",
            "213",
            "231",
            "312",
            "321",
        ]

        for i, permutation in enumerate(permutations):
            row = (i + 2) // 4
            col = (i + 2) % 4
            axes[row, col].imshow(permutation)
            axes[row, col].set_title(f"Fused image - {permutation_labels[i]}")
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()
