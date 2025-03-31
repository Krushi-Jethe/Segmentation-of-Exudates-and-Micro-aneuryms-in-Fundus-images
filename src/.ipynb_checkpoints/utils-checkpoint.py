# pylint: disable=no-name-in-module

"""
Evaluation metrics and path to the images.
"""

import os
import glob
import numpy as np
from skimage.metrics import structural_similarity as ssim


def get_data_paths()->dict:
    """
    Function to get the paths to the images.
    """

    dataset_dir = os.path.expanduser("~/Documents/datasets")
    idrid_dir = "A. Segmentation/1. Original Images"
    e_ophtha_dir = "e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset"

    #############################################
    # e_Ophtha
    ex_images = glob.glob(
        os.path.join(dataset_dir, e_ophtha_dir, "e_ophtha_EX/e_optha_EX/EX/*/*")
    )
    ma_images = glob.glob(
        os.path.join(dataset_dir, e_ophtha_dir, "e_ophtha_MA/e_optha_MA/MA/*/*")
    )
    ma_images.remove(
        os.path.join(
            dataset_dir, e_ophtha_dir, "e_ophtha_MA/e_optha_MA/MA/E0000043/Thumbs.db"
        )
    )
    ##############################################
    # Idrid
    train_imgs = glob.glob(os.path.join(dataset_dir, idrid_dir, "a. Training Set/*"))
    test_imgs = glob.glob(os.path.join(dataset_dir, idrid_dir, "b. Testing Set/*"))
    ###############################################
    # Complete dataset
    idrid_images = train_imgs + test_imgs
    e_ophtha_images = ex_images + ma_images

    return {"IDRiD": idrid_images, "e_ophtha": e_ophtha_images}


def salt_pepper_noise(img: np.ndarray, percent: float)->np.ndarray:
    """
    Generate random salt and paper noise.
    """
    array = np.zeros((img.shape[0], img.shape[1]))
    num_values_to_replace = int(percent * array.size)

    replace_indices = np.random.choice(
        array.size, num_values_to_replace * 2, replace=False
    )
    np.random.shuffle(replace_indices)

    for i in range(num_values_to_replace):
        index = replace_indices[i]
        row, col = divmod(index, array.shape[1])
        array[row, col] = -500

    for i in range(num_values_to_replace):
        index = replace_indices[num_values_to_replace + i]
        row, col = divmod(index, array.shape[1])
        array[row, col] = 500

    return array

    
class EvalMetrics:
    """
    Class to compute evaluation metrics for images.
    """

    @staticmethod
    def calculate_entropy(img: np.ndarray)->np.float64:
        """
        Compute entropy of an image.
        """
        flattened_image = img.flatten()
        histogram = np.histogram(flattened_image, bins=256, range=(0, 255))[0]
        histogram = histogram / float(np.sum(histogram))
        histogram = histogram[np.nonzero(histogram)]
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy

    @staticmethod
    def calculate_psnr(original_img: np.ndarray, processed_img: np.array)->np.float64:
        """
        Compute Peak Signal-to-Noise Ratio (PSNR).
        """
        mse = np.mean((original_img - processed_img) ** 2)
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr

    @staticmethod
    def calculate_correlation(original_img: np.ndarray, processed_img: np.array)->np.float64:
        """
        Compute Pearson Correlation Coefficient.
        """
        original_flat = original_img.flatten()
        processed_flat = processed_img.flatten()
        correlation = np.corrcoef(original_flat, processed_flat)[0, 1]
        return correlation

    @staticmethod
    def edge_preservation_index(reference_img: np.ndarray, processed_image: np.array)->np.float64:
        """
        Compute Edge Preservation Index (EPI).
        """
        numerator_sum = np.sum(np.abs(processed_image[:, 1:] - processed_image[:, :-1]))
        denominator_sum = np.sum(np.abs(reference_img[:, 1:] - reference_img[:, :-1]))
        epi = numerator_sum / denominator_sum
        return epi

    @staticmethod
    def calculate_ssim(reference_img: np.ndarray, processed_img: np.array)->np.float64:
        """
        Compute Structural Similarity Index (SSIM).
        """

        ssim_score = ssim(reference_img, processed_img)
        return ssim_score
