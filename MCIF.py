import cv2
from skimage import exposure
from skimage.filters import unsharp_mask

import numpy as np
import os
import random

from skimage.metrics import structural_similarity as ssim

from matplotlib import pyplot as plt




class get_original_data_paths():
    
    def run(self):
        
        #############################################

        #Exudates Optha
        #images
        temp=os.listdir("E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_EX/e_optha_EX/EX")
        temp=["E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_EX/e_optha_EX/EX/"+temp[i] 
              for i in range(len(temp))]
        ex_images=[i+"/"+j for i in temp for j in os.listdir(i)]

        #Micro-aneurysms Optha
        #images
        temp=os.listdir("E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_MA/e_optha_MA/MA")
        temp=["E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_MA/e_optha_MA/MA/"+temp[i] 
              for i in range(len(temp))]
        ma_images=[i+"/"+j for i in temp for j in os.listdir(i)]

        ma_images.remove('E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_MA/e_optha_MA/MA/E0000043/Thumbs.db')


        ##############################################

        #Idrid
        train_imgs = ['E:/MIT_projects/A. Segmentation/1. Original Images/a. Training Set/'+i for i in os.listdir('E:/MIT_projects/A. Segmentation/1. Original Images/a. Training Set')]
        test_imgs = ['E:/MIT_projects/A. Segmentation/1. Original Images/b. Testing Set/'+i for i in os.listdir('E:/MIT_projects/A. Segmentation/1. Original Images/b. Testing Set')]

        ###############################################

        #Complete dataset

        idrid_images = train_imgs + test_imgs
        e_ophtha_images = ex_images + ma_images

        self.images = {'idrid':idrid_images, 'e_ophtha':e_ophtha_images}


class eval_metrics():
    
    def calculate_psnr(self,original_img, processed_img):
        mse = np.mean((original_img - processed_img) ** 2)
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr

    def calculate_correlation(self,original_img, processed_img):
        original_flat = original_img.flatten()
        processed_flat = processed_img.flatten()
        correlation = np.corrcoef(original_flat, processed_flat)[0, 1]
        return correlation
    
    def edge_preservation_index(self,ref_image, processed_image):
        numerator_sum = np.sum(np.abs(processed_image[:, 1:] - processed_image[:, :-1]))
        denominator_sum = np.sum(np.abs(ref_image[:, 1:] - ref_image[:, :-1]))
        epi = numerator_sum/denominator_sum        
        return epi

    def calculate_entropy(self,image):
        flattened_image = image.flatten()
        histogram = np.histogram(flattened_image, bins=256, range=(0, 255))[0]
        histogram = histogram / float(np.sum(histogram))
        histogram = histogram[np.nonzero(histogram)]
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy

    def calculate_ssim(self,reference_img, processed_img):
        ssim_score = ssim(reference_img, processed_img)
        return ssim_score
    

class img_processor(eval_metrics):
   
    def process(self,img):
        
        img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        # 1. Gaussian filter ---> CLAHE --->  Unsharp-Mask
        gaus_img = cv2.GaussianBlur(img_gray , (5,5) , 0 , borderType = cv2.BORDER_CONSTANT)
        hist_equ = exposure.equalize_adapthist(gaus_img)
        unsharp_bilat = unsharp_mask(hist_equ , radius=7 , amount=2 )

        # 2. Split channels ---> Median Blur 2x ---> CLAHE --->  Gamma filter
        r,g,b = cv2.split(img)
        img_med = cv2.medianBlur(g,3)
        img_med_1 = cv2.medianBlur(img_med,3)
        clahe = cv2.createCLAHE(clipLimit=2,tileGridSize= (8,8))
        cl_img = clahe.apply(img_med_1)
        gamma_img = np.array(255*(cl_img/255)**0.8,dtype='uint8')


        #Merging the 3 pre-processed imgs
        l,m = gamma_img.shape
        final_img=np.ones((l,m,3))
        final_img[:,:,0]=final_img[:,:,0]*gamma_img
        final_img[:,:,1]=final_img[:,:,1]*unsharp_bilat
        final_img[:,:,2]=final_img[:,:,2]*g


        d1,d2,d3 = cv2.split(final_img)

        permutations = [
                            cv2.cvtColor(cv2.merge((d1, d2, d3)).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                            cv2.cvtColor(cv2.merge((d1, d3, d2)).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                            cv2.cvtColor(cv2.merge((d2, d1, d3)).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                            cv2.cvtColor(cv2.merge((d2, d3, d1)).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                            cv2.cvtColor(cv2.merge((d3, d1, d2)).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                            cv2.cvtColor(cv2.merge((d3, d2, d1)).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                        ]
        

        
        psnr_list = [self.calculate_psnr(img_gray, perm) for perm in permutations]
        correlation_list = [self.calculate_correlation(img_gray, perm) for perm in permutations]
        epi_list = [self.edge_preservation_index(img_gray, perm) for perm in permutations]
        entropy_list = [self.calculate_entropy(perm) for perm in permutations]
        ssim_list = [self.calculate_ssim(img_gray, perm) for perm in permutations]

 
            
        return psnr_list, correlation_list, epi_list, entropy_list, ssim_list

    
class recombination_analysis(img_processor):
    
    def __init__(self, data):
        
            self.data = data 

            if data == 'idrid':
                self.num = 54
            elif data == 'e_ophtha':
                self.num = 47
                
            self.paths = get_original_data_paths()
            self.paths.run()

    def run_experiment(self, mean=None, variance=None, percentage=None, noise_type=None):
        
        if mean==None and variance==None and percentage==None and noise_type==None:
            
            psnr_list=[]
            correlation_list=[]
            epi_list=[]
            entropy_list=[]
            ssim_list=[]

            for i in range(len(self.paths.images[self.data][:self.num])):
                    img = cv2.imread(self.paths.images[self.data][i])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    psnr_scores, correlation_scores, epi_scores, entropy_scores, ssim_scores = self.process(img)

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

            # Labels for each permutation
            permutation_labels = [
                '(d1, d2, d3)',
                '(d1, d3, d2)',
                '(d2, d1, d3)',
                '(d2, d3, d1)',
                '(d3, d1, d2)',
                '(d3, d2, d1)'
            ]

            metrics = [self.mean_psnr, self.mean_correlation, self.mean_epi, self.mean_entropy, self.mean_ssim]
            metric_names = ['Mean PSNR', 'Mean Correlation', 'Mean EPI', 'Mean Entropy', 'Mean SSIM']

            colors = ['b', 'g', 'r', 'c', 'm', 'y']

            num_permutations = len(permutation_labels)
            bar_width = 0.30
            index = np.arange(num_permutations)

            for metric, metric_name in zip(metrics, metric_names):
                plt.figure(figsize=(8, 5))

                for i in range(len(metric)):
                    plt.bar(index[i], metric[i], bar_width, color=colors[i], label=permutation_labels[i])

                plt.xlabel('Permutations')
                plt.ylabel(metric_name)
                plt.title(f'{metric_name} for Different Permutations')
                plt.xticks(index, permutation_labels)
                plt.legend()
                plt.tight_layout()
                plt.show()     

        elif noise_type=='gaussian':       

            stddev = np.sqrt(variance)
        
            psnr_list=[]
            correlation_list=[]
            epi_list=[]
            entropy_list=[]
            ssim_list=[]

            for i in range(len(self.paths.images[self.data][:self.num])):
                    img = cv2.imread(self.paths.images[self.data][i])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    gaussian_noise = np.random.normal(mean, stddev, img.shape).astype(np.uint8)
                    img = cv2.add(img, gaussian_noise)
                    
                    psnr_scores, correlation_scores, epi_scores, entropy_scores, ssim_scores = self.process(img)

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

            # Labels for each permutation
            permutation_labels = [
                '(d1, d2, d3)',
                '(d1, d3, d2)',
                '(d2, d1, d3)',
                '(d2, d3, d1)',
                '(d3, d1, d2)',
                '(d3, d2, d1)'
            ]

            metrics = [self.mean_psnr, self.mean_correlation, self.mean_epi, self.mean_entropy, self.mean_ssim]
            metric_names = ['Mean PSNR', 'Mean Correlation', 'Mean EPI', 'Mean Entropy', 'Mean SSIM']

            colors = ['b', 'g', 'r', 'c', 'm', 'y']

            num_permutations = len(permutation_labels)
            bar_width = 0.30
            index = np.arange(num_permutations)

            for metric, metric_name in zip(metrics, metric_names):
                plt.figure(figsize=(8, 5))

                for i in range(len(metric)):
                    plt.bar(index[i], metric[i], bar_width, color=colors[i], label=permutation_labels[i])

                plt.xlabel('Permutations')
                plt.ylabel(metric_name)
                plt.title(f'{metric_name} for Different Permutations')
                plt.xticks(index, permutation_labels)
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif noise_type=='salt_pepper':
            
            psnr_list=[]
            correlation_list=[]
            epi_list=[]
            entropy_list=[]
            ssim_list=[]

            for i in range(len(self.paths.images[self.data][:self.num])):
                    img = cv2.imread(self.paths.images[self.data][i])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    noise = self.salt_pepper_noise(img, percentage)
                    noise = np.expand_dims(noise, axis=-1)
                    img = np.clip(img + noise, 0, 255).astype(np.uint8)

                    
                    psnr_scores, correlation_scores, epi_scores, entropy_scores, ssim_scores = self.process(img)

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

            # Labels for each permutation
            permutation_labels = [
                '(d1, d2, d3)',
                '(d1, d3, d2)',
                '(d2, d1, d3)',
                '(d2, d3, d1)',
                '(d3, d1, d2)',
                '(d3, d2, d1)'
            ]

            metrics = [self.mean_psnr, self.mean_correlation, self.mean_epi, self.mean_entropy, self.mean_ssim]
            metric_names = ['Mean PSNR', 'Mean Correlation', 'Mean EPI', 'Mean Entropy', 'Mean SSIM']

            colors = ['b', 'g', 'r', 'c', 'm', 'y']

            num_permutations = len(permutation_labels)
            bar_width = 0.30
            index = np.arange(num_permutations)

            for metric, metric_name in zip(metrics, metric_names):
                plt.figure(figsize=(8, 5))

                for i in range(len(metric)):
                    plt.bar(index[i], metric[i], bar_width, color=colors[i], label=permutation_labels[i])

                plt.xlabel('Permutations')
                plt.ylabel(metric_name)
                plt.title(f'{metric_name} for Different Permutations')
                plt.xticks(index, permutation_labels)
                plt.legend()
                plt.tight_layout()
                plt.show() 
                
                
    def salt_pepper_noise(self, img , percent):

        array = np.zeros((img.shape[0],img.shape[1]))
        num_values_to_replace = int(percent * array.size)


        replace_indices = np.random.choice(array.size, num_values_to_replace * 2, replace=False)
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
    
        
class Visualize():
    
    def process(self,img):
        
        img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        # 1. Gaussian filter ---> CLAHE --->  Unsharp-Mask
        gaus_img = cv2.GaussianBlur(img_gray , (5,5) , 0 , borderType = cv2.BORDER_CONSTANT)
        hist_equ = exposure.equalize_adapthist(gaus_img)
        unsharp_bilat = unsharp_mask(hist_equ , radius=7 , amount=2 )

        # 2. Split channels ---> Median Blur 2x ---> CLAHE --->  Gamma filter
        r,g,b = cv2.split(img)
        img_med = cv2.medianBlur(g,3)
        img_med_1 = cv2.medianBlur(img_med,3)
        clahe = cv2.createCLAHE(clipLimit=2,tileGridSize= (8,8))
        cl_img = clahe.apply(img_med_1)
        gamma_img = np.array(255*(cl_img/255)**0.8,dtype='uint8')


        #Merging the 3 pre-processed imgs
        l,m = gamma_img.shape
        final_img=np.ones((l,m,3))
        final_img[:,:,0]=final_img[:,:,0]*gamma_img
        final_img[:,:,1]=final_img[:,:,1]*unsharp_bilat
        final_img[:,:,2]=final_img[:,:,2]*g


        d1,d2,d3 = cv2.split(final_img)

        permutations = [
                            cv2.merge((d1, d2, d3)).astype(np.uint8),
                            cv2.merge((d1, d3, d2)).astype(np.uint8),
                            cv2.merge((d2, d1, d3)).astype(np.uint8),
                            cv2.merge((d2, d3, d1)).astype(np.uint8),
                            cv2.merge((d3, d1, d2)).astype(np.uint8),
                            cv2.merge((d3, d2, d1)).astype(np.uint8)
                        ]
        return permutations
    
    def salt_pepper_noise(self, img , percent):

        array = np.zeros((img.shape[0],img.shape[1]))
        num_values_to_replace = int(percent * array.size)


        replace_indices = np.random.choice(array.size, num_values_to_replace * 2, replace=False)
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
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        permutations = self.process(img)
        
        self.plot(img, permutations)
        
        if noise_type=='gaussian':
            stddev = np.sqrt(variance)

            gaussian_noise = np.random.normal(mean, stddev, img.shape).astype(np.uint8)
            gaussian_img = cv2.add(img, gaussian_noise)

            gaussian_permutations = self.process(gaussian_img)
            
            self.plot(gaussian_img, gaussian_permutations)
        
        if noise_type=='salt_pepper':
            noise = self.salt_pepper_noise(img, percentage)
            noise = np.expand_dims(noise, axis=-1)
            salt_pepper_img = np.clip(img + noise, 0, 255).astype(np.uint8)

            salt_pepper_permutations = self.process(salt_pepper_img)
            
            self.plot(salt_pepper_img, salt_pepper_permutations)
        
        
        
        
        
    def plot(self, original, permutations):
        
        num_rows = len(permutations) // 3 + 1
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))

        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image')

        for i, permutation in enumerate(permutations):
            row = (i + 1) // 3
            col = (i + 1) % 3
            axes[row, col].imshow(permutation, cmap='gray')
            axes[row, col].set_title(f'Perm. {i+1}')
            axes[row, col].axis('off')  # Turn off axis labels and ticks

        plt.tight_layout()
        plt.show()
