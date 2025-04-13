"""
Main
"""

import random
import numpy as np
from tqdm import tqdm
from utils import get_data_paths
from .processor import ImageProcessor

SEED = 108

random.seed(SEED)
np.random.seed(SEED)

def main():
    """
    Docstring Placeholder.
    """
    data_paths = get_data_paths()
    img_processor = ImageProcessor()

    for dataset, paths in data_paths.items():
        metrics_dict = {
            "psnr": [],
            "corr": [],
            "epi": [],
            "ssim": [],
        }
        for path in tqdm(paths, total=len(paths)):
            img_processor.run(path)

            for key, value in metrics_dict.items():
                value.append(img_processor.metrics[key])

        metrics_dict = {key: np.array(value) for key, value in metrics_dict.items()}
        metrics_mean_dict = {
            "mean_" + key: np.mean(value, axis=0) for key, value in metrics_dict.items()
        }

        print("\n####################################")
        print(f"Printing metrics for {dataset} dataset")
        for key, value in metrics_mean_dict.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
