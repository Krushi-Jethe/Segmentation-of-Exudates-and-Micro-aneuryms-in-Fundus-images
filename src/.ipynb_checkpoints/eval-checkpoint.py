"""
Evaluation metrics
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


class EvalMetrics:
    """
    Class to test.
    """

    def calculate_psnr(self, original_img, processed_img):
        """
        Function to test.
        """
        mse = np.mean((original_img - processed_img) ** 2)
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr

    def calculate_correlation(self, original_img, processed_img):
        """
        Function to test.
        """
        original_flat = original_img.flatten()
        processed_flat = processed_img.flatten()
        correlation = np.corrcoef(original_flat, processed_flat)[0, 1]
        return correlation

    def edge_preservation_index(self, ref_image, processed_image):
        """
        Function to test.
        """
        numerator_sum = np.sum(np.abs(processed_image[:, 1:] - processed_image[:, :-1]))
        denominator_sum = np.sum(np.abs(ref_image[:, 1:] - ref_image[:, :-1]))
        epi = numerator_sum / denominator_sum
        return epi

    def calculate_entropy(self, image):
        """
        Function to test.
        """
        flattened_image = image.flatten()
        histogram = np.histogram(flattened_image, bins=256, range=(0, 255))[0]
        histogram = histogram / float(np.sum(histogram))
        histogram = histogram[np.nonzero(histogram)]
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy

    def calculate_ssim(self, reference_img, processed_img):
        """
        Function to test.
        """

        ssim_score = ssim(reference_img, processed_img)
        return ssim_score