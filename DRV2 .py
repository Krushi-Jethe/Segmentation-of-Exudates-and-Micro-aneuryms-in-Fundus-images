import os
import numpy as np
import cv2
import albumentations as A
import random

class get_original_data_paths():
    def __init__(self):
        self.e_ophtha = {
            'train': {
                'EX': {'images': [], 'masks': []},
                'MA': {'images': [], 'masks': []}
            },
            'val': {
                'EX': {'images': [], 'masks': []},
                'MA': {'images': [], 'masks': []}
            },
            'test': {
                'EX': {'images': [], 'masks': []},
                'MA': {'images': [], 'masks': []}
            }
        }
        
        self.idrid = {
            'train': {
                'EX': {'images': [], 'masks': []},
                'MA': {'images': [], 'masks': []}
            },
            'val': {
                'EX': {'images': [], 'masks': []},
                'MA': {'images': [], 'masks': []}
            },
            'test': {
                'EX': {'images': [], 'masks': []},
                'MA': {'images': [], 'masks': []}
            }
        }
    
    def run(self):

                #Exudates Optha

                #images
                temp=os.listdir("E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_EX/e_optha_EX/EX")
                temp=["E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_EX/e_optha_EX/EX/"+temp[i] 
                      for i in range(len(temp))]
                ex_images=[i+"/"+j for i in temp for j in os.listdir(i)]

                #masks
                temp=os.listdir("E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_EX/e_optha_EX/Annotation_EX")
                temp=["E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_EX/e_optha_EX/Annotation_EX/"+temp[i] 
                      for i in range(len(temp))]
                ex_masks=[i+"/"+j for i in temp for j in os.listdir(i)]

                ex_masks.remove('E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_EX/e_optha_EX/Annotation_EX/E0000404/Thumbs.db')

                self.e_ophtha['train']['EX'] = {'images':ex_images[:30] , 'masks':ex_masks[:30]} 
                self.e_ophtha['val']['EX'] = {'images':ex_images[30:37] , 'masks':ex_masks[30:37]} 
                self.e_ophtha['test']['EX'] = {'images':ex_images[37:] , 'masks':ex_masks[37:]}

                #Micro-aneurysms Optha

                #images
                temp=os.listdir("E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_MA/e_optha_MA/MA")
                temp=["E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_MA/e_optha_MA/MA/"+temp[i] 
                      for i in range(len(temp))]
                ma_images=[i+"/"+j for i in temp for j in os.listdir(i)]

                
                #masks
                temp=os.listdir("E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_MA/e_optha_MA/Annotation_MA")
                temp=["E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_MA/e_optha_MA/Annotation_MA/"+temp[i] 
                      for i in range(len(temp))]
                ma_masks=[i+"/"+j for i in temp for j in os.listdir(i)]

                ma_images.remove('E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_MA/e_optha_MA/MA/E0000043/Thumbs.db')
                ma_masks.remove('E:/MIT_projects/Diabetic retinopathy/e_ophtha_dataset-20230303T080518Z-001/e_ophtha_dataset/e_ophtha_MA/e_optha_MA/Annotation_MA/E0000043/Thumbs.db')
                
                self.e_ophtha['train']['MA'] = {'images':ma_images[:104] , 'masks':ma_masks[:104]}
                self.e_ophtha['val']['MA'] = {'images':ma_images[104:118] , 'masks':ma_masks[104:118]}
                self.e_ophtha['test']['MA'] = {'images':ma_images[118:] , 'masks':ma_masks[118:]}

                ##############################################

                #Idrid
                train_imgs = ['E:/MIT_projects/A. Segmentation/1. Original Images/a. Training Set/'+i for i in os.listdir('E:/MIT_projects/A. Segmentation/1. Original Images/a. Training Set')]
                test_imgs = ['E:/MIT_projects/A. Segmentation/1. Original Images/b. Testing Set/'+i for i in os.listdir('E:/MIT_projects/A. Segmentation/1. Original Images/b. Testing Set')]

                train_ma_masks=["E:/MIT_projects/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms/" + i for  i in os.listdir("E:/MIT_projects/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms")]
                test_ma_masks=["E:/MIT_projects/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms/" + i for i in os.listdir("E:/MIT_projects/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms")]


                train_ex_masks=["E:/MIT_projects/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates/" + i for  i in os.listdir("E:/MIT_projects/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates")]
                test_ex_masks=["E:/MIT_projects/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/3. Hard Exudates/" + i for i in os.listdir("E:/MIT_projects/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/3. Hard Exudates")]

                self.idrid['train']['EX'] = {'images':train_imgs[:44] , 'masks':train_ex_masks[:44]} 
                self.idrid['val']['EX'] = {'images':train_imgs[44:] , 'masks':train_ex_masks[44:]}
                self.idrid['test']['EX'] = {'images':test_imgs , 'masks':test_ex_masks}
                
                self.idrid['train']['MA'] = {'images':train_imgs[:44] , 'masks':train_ma_masks[:44]}
                self.idrid['val']['MA'] = {'images':train_imgs[44:] , 'masks':train_ma_masks[44:]}
                self.idrid['test']['MA'] = {'images':test_imgs , 'masks':test_ma_masks}

                ###############################################


# Creating the X_train and Y_train using the paths
class create_data_array():
        def create_inputs(self,path_images,path_masks):
            X_train = np.zeros((len(path_images),256,256,3),dtype=np.uint8)
            Y_train = np.zeros((len(path_masks),256,256),dtype=bool)

            for i in range(len(path_images)):
                img = cv2.imread( path_images[i] , 1 )
                img = cv2.cvtColor( img , cv2.COLOR_BGR2RGB )
                img = cv2.resize( img , (256,256) )
                X_train[i] = img

                mask = cv2.imread( path_masks[i] , 0 )
                mask = cv2.resize( mask , (256,256) )
                Y_train[i] = mask

            return X_train , Y_train
 
#perform augmentation and increase data
class increase_data():
    
        def __init__(self , X_train , Y_train , num_aug):
                self.X_train = X_train
                self.Y_train = Y_train
                self.num_aug = num_aug - len(X_train)

        def augment_images(self):

                x_array = np.zeros((self.num_aug,256,256,3),dtype=np.uint8)
                y_array = np.zeros((self.num_aug,256,256)  ,dtype=np.uint8)

                data_size = len(self.X_train)

                #Defining transforms to use
                transform = A.Compose([
                                    A.HorizontalFlip(p=0.6),
                                    A.RandomBrightnessContrast(p=0.3),
                                    A.VerticalFlip(p=0.6),
                                    A.RandomRotate90(p=0.5),
                                    A.Transpose(p=0.2),
                                    A.Rotate(always_apply=False, p=0.5, limit=(-45, 45), interpolation=0, border_mode=0, 
                                                           value=(0, 0, 0), mask_value=None, rotate_method='largest_box',crop_border=False),
                                    A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=False, elementwise=False, p=1.0),

                                    ])
                #Loop for the number of augmentations
                for i in range(self.num_aug):

                    #Sample randomly from the class data
                    random_sample_num = random.sample(range(data_size),1)

                    img = self.X_train[random_sample_num].reshape((256,256,3))
                    mask = self.Y_train[random_sample_num].reshape((256,256)).astype(np.uint8)

                    transformed = transform(image = img , mask = mask)

                    transformed_image = transformed['image']
                    transformed_mask = transformed['mask']

                    x_array[i]=transformed_image
                    y_array[i]=transformed_mask
                    
                self.X_aug = x_array
                self.Y_aug = y_array


                return x_array , y_array

        def add_aug_data(self):

                self.X_train = np.append(self.X_train , self.X_aug , axis=0)
                self.Y_train = np.append(self.Y_train , self.Y_aug , axis=0)

                #Shuffling the data
                indexes = np.arange(len(self.X_train))
                np.random.shuffle(indexes)

                self.X_train = self.X_train[indexes]
                self.Y_train = self.Y_train[indexes]*255

                return self.X_train , self.Y_train
            
        def write_to_dir(self,path):
            
            os.makedirs(os.path.join(path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(path, 'masks'), exist_ok=True)
            
            for i in range(len(self.X_train)):
            
                bgr_image = cv2.cvtColor(self.X_train[i], cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(path, 'images', f'{i}.jpg'), bgr_image)
                cv2.imwrite(os.path.join(path, 'masks', f'{i}.jpg'), self.Y_train[i])
                

                
        def save_as_numpy(self,path):
            
                 np.save(path + '/' + 'X', self.X_train)
                 np.save(path + '/' + 'Y', self.Y_train)
                 
                 
                 
                 
                 
from keras_unet_collection import models,losses
from tensorflow.keras.metrics import MeanIoU 
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
from datetime import datetime 
import tensorflow as tf
                    
class config_exp():
    '''
    SAMPLE CONFIGURATION TO BE PASSED FOR TRAINING

    config = {
            'data': 'Original'  -- (Original or preprocessed)
            'batch_size':16,
            'backbone':'ResNet50', 
            'batch_norm':True,
            'loss': unified_focal_loss,
            'lr':1e-4,
            'epochs':100,
            'callbacks': [early_stopping , reduce_lr_on_plateau , None]
            }
            
    exp_1 = config_exp(**config_1)
    exp_1.run(X_train, Y_train,(X_val,Y_val))
    exp_1.plot_pred(X_test, Y_test, 'idrid') 
    exp_1.calc_IoU(X_test, Y_test, 'idrid')
    exp_1.calc_classification_report(X_test, Y_test, 'idrid')
    
    '''
    def __init__(self, data, batch_size, backbone, batch_norm, loss, lr, epochs, callbacks):

        
        self.data = data
        self.batch_size = batch_size
        self.backbone = backbone
        self.batch_norm = batch_norm
        self.loss = loss
        self.lr = lr
        self.epochs = epochs
        self.dir_ = f'exp_batch_size_{str(batch_size)}_{backbone}_{loss}_lr_{str(lr)}_epochs_{str(epochs)}'
        os.mkdir('/content/drive/MyDrive/DRV2/' + self.data + '/experiments/'+self.dir_)
        os.mkdir('/content/drive/MyDrive/DRV2/' + self.data + '/experiments/'+self.dir_+'/idrid')
        os.mkdir('/content/drive/MyDrive/DRV2/' + self.data + '/experiments/'+self.dir_+'/e_ophtha')
        self.callbacks = [tf.keras.callbacks.ModelCheckpoint(
                                               filepath='/content/drive/MyDrive/DRV2/'+ self.data+'/experiments/'+self.dir_ +'/model.keras', 
                                               monitor='val_dice_coef', 
                                               verbose=1, 
                                               save_best_only=True, 
                                               mode='max')]
        
        for i in callbacks:
              
                if i == 'early_stopping':
                    
                         self.callbacks= self.callbacks+[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                    mode='auto',
                                    verbose=1,
                                    patience=5)]
                        
                if i == 'reduce_lr_on_plateau':
                    
                         self.callbacks= self.callbacks+[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                    patience=3,
                                    verbose=1,
                                    factor=.5, 
                                    min_lr=0.0000001)]
        
        
    def run(self, X_train, Y_train, val_dataset):
        model = models.att_unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                   n_labels=1, 
                   stack_num_down=2, stack_num_up=2, 
                   activation='ReLU', 
                   atten_activation='ReLU', attention='add', 
                   output_activation='Sigmoid', 
                   batch_norm=self.batch_norm, pool=False, unpool=False, 
                   backbone=self.backbone, weights='imagenet', 
                   freeze_backbone=True, freeze_batch_norm=True, 
                   name='attunet')

        model.compile(loss=self.loss, optimizer=Adam(learning_rate=self.lr), 
                      metrics=['accuracy', losses.dice_coef])

        start3 = datetime.now()

        history = model.fit(X_train,Y_train, 
                            verbose=1,
                            batch_size=self.batch_size,
                            validation_data=val_dataset,
                            shuffle=False,
                            epochs=self.epochs,
                           callbacks = self.callbacks)

        stop3 = datetime.now()

        # Execution time of the model 
        execution_time_att_Unet = stop3 - start3
        print("Attention UNet execution time is: ", execution_time_att_Unet)


        # Define the path where you want to save the pickled file
        history_path = '/content/drive/MyDrive/DRV2/' + self.data + '/experiments/' + self.dir_ + '/history.pkl'

        # Save the 'history' dictionary as a pickled file
        with open(history_path, 'wb') as history_file:
            pickle.dump(history.history, history_file)
        
        self.model = model
        self.history = history
 
    def plot_loss_acc(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        acc = self.history.history['dice_coef']
        val_acc = self.history.history['val_dice_coef']


        plt.plot(epochs, acc, 'y', label='Training Dice')
        plt.plot(epochs, val_acc, 'r', label='Validation Dice')
        plt.title('Training and validation Dice')
        plt.xlabel('Epochs')
        plt.ylabel('Dice')
        plt.legend()
        plt.show()
            
            
    def plot_pred(self, X_test, Y_test, data):
            print(f'Printing prediction masks of {data} data')
            for test_img_number in range(0,X_test.shape[0]-1):
                    test_img = X_test[test_img_number]
                    ground_truth = Y_test[test_img_number]
                    test_img_input=np.expand_dims(test_img, 0)
                    prediction = (self.model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

                    plt.figure(figsize=(16, 8))
                    plt.subplot(231)
                    plt.title('Testing Image')
                    plt.imshow(test_img, cmap='gray')
                    plt.subplot(232)
                    plt.title('Testing Label')
                    plt.imshow(ground_truth[:,:], cmap='gray')
                    plt.subplot(233)
                    plt.title('Prediction on test image')
                    plt.imshow(prediction, cmap='gray')


                    plt.show()
                    
                    
    def calc_IoU(self, X_test, Y_test, data):
            n_classes = 2
            IoU_values = []

            for img in range(0, X_test.shape[0]):
                temp_img = X_test[img]
                ground_truth = Y_test[img]
                temp_img_input = np.expand_dims(temp_img, 0)
                prediction = (self.model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

                IoU = MeanIoU(num_classes=n_classes)
                IoU.update_state(ground_truth[:,:], prediction)
                IoU = IoU.result().numpy()
                IoU_values.append(IoU)

            df = pd.DataFrame(IoU_values, columns=["IoU"])
            df = df[df.IoU != 1.0]
            mean_IoU = df.mean().values
            print("Mean IoU is: ", mean_IoU)
            
            if data=='idrid':
                df.to_csv('/content/drive/MyDrive/DRV2/' + self.data + '/experiments/'+self.dir_+'/idrid/iou.csv')
            elif data=='e_ophtha':
                df.to_csv('/content/drive/MyDrive/DRV2/' + self.data + '/experiments/'+self.dir_+'/e_ophtha/iou.csv')
                
            
            
    def calc_classification_report(self, X_test, Y_test, data):
            flat_ground_truth = Y_test.reshape(-1)
            flat_prediction = (self.model.predict(X_test)[:, :, :, 0] > 0.5).astype(np.uint8).reshape(-1)

            report = classification_report(flat_ground_truth, flat_prediction)
            
            if data=='idrid':
                report_path = '/content/drive/MyDrive/DRV2/' + self.data + '/experiments/'+self.dir_+'/idrid/classification_report.txt'
            elif data=='e_ophtha':
                report_path = '/content/drive/MyDrive/DRV2/' + self.data + '/experiments/'+self.dir_+'/e_ophtha/classification_report.txt'
            
            with open(report_path, 'w') as report_file:
                report_file.write(report)
                
            print(report)
