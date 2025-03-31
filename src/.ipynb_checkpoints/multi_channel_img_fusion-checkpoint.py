# pylint: disable=no-member, too-many-positional-arguments, too-many-arguments

"""
Script
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import get_data_paths, salt_pepper_noise
from processor import ImageProcessor


class RecombinationAnalysis(ImageProcessor):
    """
    Class to test.
    """

    def __init__(self, data: str):

        self.data = data

        if data == "idrid":
            self.num = 54
        elif data == "e_ophtha":
            self.num = 47

        self.paths = GetDataPaths()
        self.paths.run()

        self.psnr_arr = None
        self.mean_psnr = None
        self.correlation_arr = None
        self.mean_correlation = None
        self.epi_arr = None
        self.mean_epi = None
        self.entropy_arr = None
        self.mean_entropy = None
        self.ssim_arr = None
        self.mean_ssim = None

    def run_experiment(
        self,
        mean: float = None,
        variance: float = None,
        percentage: float = None,
        noise_type: str = None,
    ):
        """
        Function to test.
        """

        if (
            mean is None
            and variance is None
            and percentage is None
            and noise_type is None
        ):

            psnr_list = []
            correlation_list = []
            epi_list = []
            entropy_list = []
            ssim_list = []

            for i in range(len(self.paths.images[self.data][: self.num])):

                img = cv2.imread(self.paths.images[self.data][i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if noise_type:
                    if noise_type == "salt_pepper":
                        noise = self.salt_pepper_noise(img, percentage)
                        noise = np.expand_dims(noise, axis=-1)
                        img = np.clip(img + noise, 0, 255).astype(np.uint8)
                    elif noise_type == "gausian":
                        stddev = np.sqrt(variance)
                        gaussian_noise = np.random.normal(
                            mean, stddev, img.shape
                        ).astype(np.uint8)
                        img = cv2.add(img, gaussian_noise)
                    else:
                        raise ValueError(
                            f"Invalid noise_type: {noise_type}. Parameter noise_type must be `gaussian` or `salt_pepper`"
                        )

                (
                    psnr_scores,
                    correlation_scores,
                    epi_scores,
                    entropy_scores,
                    ssim_scores,
                ) = self.process(img)

                psnr_list.append(psnr_scores)
                correlation_list.append(correlation_scores)
                epi_list.append(epi_scores)
                entropy_list.append(entropy_scores)
                ssim_list.append(ssim_scores)

            self.psnr_arr = np.array(psnr_list)
            self.mean_psnr = np.mean(self.psnr_arr, axis=0)

            self.correlation_arr = np.array(correlation_list)
            self.mean_correlation = np.mean(self.correlation_arr, axis=0)

            self.epi_arr = np.array(epi_list)
            self.mean_epi = np.mean(self.epi_arr, axis=0)

            self.entropy_arr = np.array(entropy_list)
            self.mean_entropy = np.mean(self.entropy_arr, axis=0)

            self.ssim_arr = np.array(ssim_list)
            self.mean_ssim = np.mean(self.ssim_arr, axis=0)

            print("Mean PSNR:", self.mean_psnr)
            print("Mean Correlation:", self.mean_correlation)
            print("Mean EPI:", self.mean_epi)
            print("Mean Entropy:", self.mean_entropy)
            print("Mean SSIM:", self.mean_ssim)

    def create_bar_plot(self):
        """
        Bar plot.
        """
        # Labels for each permutation
        permutation_labels = [
            "(d1, d2, d3)",
            "(d1, d3, d2)",
            "(d2, d1, d3)",
            "(d2, d3, d1)",
            "(d3, d1, d2)",
            "(d3, d2, d1)",
        ]

        metrics = [
            self.mean_psnr,
            self.mean_correlation,
            self.mean_epi,
            self.mean_entropy,
            self.mean_ssim,
        ]

        metric_names = [
            "Mean PSNR",
            "Mean Correlation",
            "Mean EPI",
            "Mean Entropy",
            "Mean SSIM",
        ]

        colors = ["b", "g", "r", "c", "m", "y"]

        num_permutations = len(permutation_labels)
        bar_width = 0.30
        index = np.arange(num_permutations)

        for metric, metric_name in zip(metrics, metric_names):
            plt.figure(figsize=(8, 5))

            for i, _ in enumerate(metric):

                plt.bar(
                    index[i],
                    metric[i],
                    bar_width,
                    color=colors[i],
                    label=permutation_labels[i],
                )

            plt.xlabel("Permutations")
            plt.ylabel(metric_name)
            plt.title(f"{metric_name} for Different Permutations")
            plt.xticks(index, permutation_labels)
            plt.legend()
            plt.tight_layout()
            plt.show()
