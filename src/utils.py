# pylint: disable=no-name-in-module, too-many-locals

"""
Evaluation metrics, path to the images and noise generation.
"""
import os
import glob
import random
import copy
import numpy as np
from skimage.metrics import structural_similarity as ssim

SEED = 108

random.seed(SEED)
np.random.seed(SEED)


def get_data_paths(objective: str) -> dict:
    """
    Retrieve paths to the images for the datasets IDRiD and e-ophtha
    and store them in a dictionary.

    Returns:
        dict: Containing keys "IDRiD" and "e_ophtha" and values as list of paths.
    """

    dataset_dir = os.path.expanduser("/mnt/d/MIT_projects")
    idrid_dir = "A. Segmentation"
    e_ophtha_dir = "e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset"
    template = {
        "train": {"EX": {"images": [], "masks": []}, "MA": {"images": [], "masks": []}},
        "val": {"EX": {"images": [], "masks": []}, "MA": {"images": [], "masks": []}},
        "test": {"EX": {"images": [], "masks": []}, "MA": {"images": [], "masks": []}},
    }

    #############################################
    # e_Ophtha
    e_ophtha = copy.deepcopy(template)

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

    ex_masks = glob.glob(
        os.path.join(
            dataset_dir, e_ophtha_dir, "e_ophtha_EX/e_optha_EX/Annotation_EX/*/*"
        )
    )

    ma_masks = glob.glob(
        os.path.join(
            dataset_dir, e_ophtha_dir, "e_ophtha_MA/e_optha_MA/Annotation_MA/*/*"
        )
    )
    ma_masks.remove(
        os.path.join(
            dataset_dir,
            e_ophtha_dir,
            "e_ophtha_MA/e_optha_MA/Annotation_MA/E0000043/Thumbs.db",
        )
    )

    e_ophtha["train"]["EX"] = {"images": ex_images[:30], "masks": ex_masks[:30]}
    e_ophtha["val"]["EX"] = {"images": ex_images[30:37], "masks": ex_masks[30:37]}
    e_ophtha["test"]["EX"] = {"images": ex_images[37:], "masks": ex_masks[37:]}

    e_ophtha["train"]["MA"] = {"images": ma_images[:104], "masks": ma_masks[:104]}
    e_ophtha["val"]["MA"] = {"images": ma_images[104:118], "masks": ma_masks[104:118]}
    e_ophtha["test"]["MA"] = {"images": ma_images[118:], "masks": ma_masks[118:]}
    ##############################################
    # Idrid
    IDRiD = copy.deepcopy(template)

    train_imgs = glob.glob(
        os.path.join(dataset_dir, idrid_dir, "1. Original Images/a. Training Set/*")
    )
    test_imgs = glob.glob(
        os.path.join(dataset_dir, idrid_dir, "1. Original Images/b. Testing Set/*")
    )

    train_masks_ma = glob.glob(
        os.path.join(
            dataset_dir,
            idrid_dir,
            "2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms/*",
        )
    )
    test_masks_ma = glob.glob(
        os.path.join(
            dataset_dir,
            idrid_dir,
            "2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms/*",
        )
    )

    train_masks_ex = glob.glob(
        os.path.join(
            dataset_dir,
            idrid_dir,
            "2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates/*",
        )
    )
    test_masks_ex = glob.glob(
        os.path.join(
            dataset_dir,
            idrid_dir,
            "2. All Segmentation Groundtruths/b. Testing Set/3. Hard Exudates/*",
        )
    )

    IDRiD["train"]["EX"] = {"images": train_imgs[:44], "masks": train_masks_ex[:44]}
    IDRiD["val"]["EX"] = {"images": train_imgs[44:], "masks": train_masks_ex[44:]}
    IDRiD["test"]["EX"] = {"images": test_imgs, "masks": test_masks_ex}

    IDRiD["train"]["MA"] = {"images": train_imgs[:44], "masks": train_masks_ma[:44]}
    IDRiD["val"]["MA"] = {"images": train_imgs[44:], "masks": train_masks_ma[44:]}
    IDRiD["test"]["MA"] = {"images": test_imgs, "masks": test_masks_ma}

    ###############################################
    # Complete dataset
    idrid_images = train_imgs + test_imgs
    e_ophtha_images = ex_images + ma_images

    if objective == "processing":
        return {"IDRiD": idrid_images, "e_ophtha": e_ophtha_images}

    if objective == "training":
        return {"IDRiD": IDRiD, "e_ophtha": e_ophtha}

    raise ValueError(
        f"Incorrect value for param objective - {objective} can only be `processing` or `training`"
    )


def salt_pepper_noise(img: np.ndarray, percent: float) -> np.ndarray:
    """
    Given an input image and percentage of noise, the function generates
    salt and pepper noise to be injected to the image.

    Args:
        img (np.ndarray): Image to inject noise into.
        percent (float): Percentage of noise to be injected. E.g.: 0.10 for 10%

    Returns:
        np.ndarray: Salt and pepper noise array which can be added to the image.
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
    def calculate_entropy(img: np.ndarray) -> np.float64:
        """
        Calculates entropy of the image.

        Args:
            img (np.ndarray): Gray scale image.

        Returns:
            np.float64: Entropy
        """

        flattened_image = img.flatten()
        histogram = np.histogram(flattened_image, bins=256, range=(0, 255))[0]
        histogram = histogram / float(np.sum(histogram))
        histogram = histogram[np.nonzero(histogram)]
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy

    @staticmethod
    def calculate_psnr(original_img: np.ndarray, processed_img: np.array) -> np.float64:
        """
        Calculates peak signal to noise ratio between original image and
        processed image.

        Args:
            original_img (np.ndarray): Original gray scale image.
            processed_img (np.array): Processed gray scale image.

        Returns:
            np.float64: PSNR
        """

        mse = np.mean((original_img - processed_img) ** 2)
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr

    @staticmethod
    def calculate_correlation(
        original_img: np.ndarray, processed_img: np.array
    ) -> np.float64:
        """
        Calculates correlation between original image and
        processed image.

        Args:
            original_img (np.ndarray): Original gray scale image.
            processed_img (np.array): Processed gray scale image.

        Returns:
            np.float64: Correlation
        """

        original_flat = original_img.flatten()
        processed_flat = processed_img.flatten()
        correlation = np.corrcoef(original_flat, processed_flat)[0, 1]
        return correlation

    @staticmethod
    def edge_preservation_index(
        reference_img: np.ndarray, processed_image: np.array
    ) -> np.float64:
        """
        Calculates edge preservation index between original image and
        processed image.

        Args:
            reference_img (np.ndarray): Original gray scale image.
            processed_image (np.array): Processed gray scale image.

        Returns:
            np.float64: _description_
        """

        numerator_sum = np.sum(np.abs(processed_image[:, 1:] - processed_image[:, :-1]))
        denominator_sum = np.sum(np.abs(reference_img[:, 1:] - reference_img[:, :-1]))
        epi = numerator_sum / denominator_sum
        return epi

    @staticmethod
    def calculate_ssim(
        reference_img: np.ndarray, processed_img: np.array
    ) -> np.float64:
        """
        Calculates structural similarity index measure between original
        image and processed image.

        Args:
            reference_img (np.ndarray): Original gray scale image.
            processed_img (np.array): Processed gray scale image.

        Returns:
            np.float64: SSIM
        """

        ssim_score = ssim(reference_img, processed_img)
        return ssim_score
