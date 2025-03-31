"""
Visualisation
"""

class Visualize:
    """
    Class to test.
    """

    def process(self, img):
        """
        Function to test.
        """

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 1. Gaussian filter ---> CLAHE --->  Unsharp-Mask
        gaus_img = cv2.GaussianBlur(img_gray, (5, 5), 0, borderType=cv2.BORDER_CONSTANT)
        hist_equ = exposure.equalize_adapthist(gaus_img)
        unsharp_bilat = unsharp_mask(hist_equ, radius=7, amount=2)

        # 2. Split channels ---> Median Blur 2x ---> CLAHE --->  Gamma filter
        r, g, b = cv2.split(img)
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
        return permutations

    def salt_pepper_noise(self, img, percent):
        """
        Function to test.
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

    def viz(self, img, mean=None, variance=None, percentage=None, noise_type=None):
        """
        Function to test.
        """

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        permutations = self.process(img)

        self.plot(img, permutations)

        if noise_type == "gaussian":
            stddev = np.sqrt(variance)

            gaussian_noise = np.random.normal(mean, stddev, img.shape).astype(np.uint8)
            gaussian_img = cv2.add(img, gaussian_noise)

            gaussian_permutations = self.process(gaussian_img)

            self.plot(gaussian_img, gaussian_permutations)

        if noise_type == "salt_pepper":
            noise = self.salt_pepper_noise(img, percentage)
            noise = np.expand_dims(noise, axis=-1)
            salt_pepper_img = np.clip(img + noise, 0, 255).astype(np.uint8)

            salt_pepper_permutations = self.process(salt_pepper_img)

            self.plot(salt_pepper_img, salt_pepper_permutations)

    def plot(self, original, permutations):
        """
        Function to test.
        """

        num_rows = len(permutations) // 3 + 1
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

        axes[0, 0].imshow(original)
        axes[0, 0].set_title("Original Image")

        for i, permutation in enumerate(permutations):
            row = (i + 1) // 3
            col = (i + 1) % 3
            axes[row, col].imshow(permutation, cmap="gray")
            axes[row, col].set_title(f"Perm. {i+1}")
            axes[row, col].axis("off")  # Turn off axis labels and ticks

        plt.tight_layout()
        plt.show()
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