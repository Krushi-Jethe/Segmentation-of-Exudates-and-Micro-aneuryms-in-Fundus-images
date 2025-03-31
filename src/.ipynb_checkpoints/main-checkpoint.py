"""
Main
"""

import numpy as np
from tqdm import tqdm
from utils import get_data_paths
from processor import ImageProcessor

data_paths = get_data_paths()
img_processor = ImageProcessor()


for dataset, paths in data_paths.items():
    if dataset == "e_ophtha":
        metrics_dict = {
        "psnr": [],
        "corr": [],
        "epi": [],
        "entropy": [],
        "ssim": [],
    }
        for idx, path in tqdm(enumerate(paths), total=len(paths)):
            img_processor.run(path, "salt_pepper")
    
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
            