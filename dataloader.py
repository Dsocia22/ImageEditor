# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:59:03 2021

@author: Alex Trehubenko
"""

import os
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, RandomCrop
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
import glob

class ImageDataset(Dataset):
    def __init__(self, img_dir, split=None, retain = None, image_format='JPG', transform=None):
        '''
        Initialize the image dataset. 

        Parameters
        ----------
        img_dir : string
            The root directory of the data folder.
        split : pandas series
            A series that indicates which indices belong to each split
        retain: int
            Which values of split should stay in this dataset.
        image_format : string, optional
            The image format, used to point the dataset constructor at the correct folder. The default is 'JPEG'.
        transform : function, optional
            A transformation for the orignial image.

        Returns
        -------
        None.

        '''
        # Pattern to detect which file names match the desired format.
        label_list = []
        for expert in ['A','B','C','D','E']:
            glob_pattern = 'Expert '+expert+'/'+image_format.upper()+'/*.'+image_format.lower()
            # Get a series of file names matching the glob format.
            img_label_expert = pd.Series(glob.glob(os.path.join(img_dir,glob_pattern)))
            # If the dataset has some data held out, retain values where series split == True. Reset index. 
            if split is not None:
                img_label_expert = img_label_expert[0:len(split)][split == retain].reset_index(drop = True)
            label_list.append(img_label_expert)
        
        self.img_labels = list(pd.concat(label_list))
        self.img_dir = img_dir
        # Note the image format, png, jpeg, ect..
        self.image_format = image_format
        # Action to transform both the input and output tensors. 
        self.transform = transform

    def __len__(self):
        '''
        Gets the number of datapoints in the dataset

        Returns
        -------
        TYPE : int
            Number of image labels in the dataset.

        '''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''
        

        Parameters
        ----------
        idx : int
            index of a particular piece of data.
        Returns
        -------
        sample : TYPE
            DESCRIPTION.

        '''
        # Get the path by index
        edited_image_path = self.img_labels[idx]
        # Read the image by path
        edited_image = read_image(edited_image_path)/255
        # Split path, just get image name 
        file_name = os.path.split(edited_image_path)[1]
        # Construct the path to the original image
        original_image_path = os.path.join(self.img_dir,'Original',self.image_format,file_name)
        # Read the original image
        original_image = read_image(original_image_path)/255
        
        if self.transform is not None:
            # Perform the same random transformations on the input and output tensors.
            # The main reason seed is set is so cropping is consistent between images.  
            seed = np.random.randint(10**5)
            torch.manual_seed(seed)
            original_image = self.transform(original_image)
            torch.manual_seed(seed)
            edited_image = self.transform(edited_image)
            
        sample = {"original_image": original_image, "edited_image": edited_image}
        return sample


def generate_test_train_dataloader(image_dir, batch_size, num_workers, test_split=0.2, val_split=0.2, img_size=(100, 100),number_images = 5000):
    # Get splits. Form a series of 0, 1, or 2 indicating which split an original image should belong to.
    test_count = round(number_images * test_split)
    val_count = round(number_images * val_split)
    train_count = number_images - test_count - val_count
    # 0 = training data, 1 = validation data, 2 = testing data
    splits = pd.Series([0] * train_count + [1] * val_count + [2] * test_count).sample(frac=1).reset_index(drop=True)

    # Create dataloaders
    train_dataset = ImageDataset(image_dir, split=splits, retain = 0, transform=RandomCrop(img_size))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_dataset = ImageDataset(image_dir, split=splits, retain = 1, transform=RandomCrop(img_size))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_dataset = ImageDataset(image_dir, split=splits, retain = 2, transform=RandomCrop(img_size))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


    return train_dataloader, val_dataloader, test_dataloader
    
def perform_tests_dataset(dataset,expected_len = None, idx_to_get = 0):
    '''
    Performs tests on the dataset object to ensure it is implemented as intended.

    Parameters
    ----------
    dataset : ImageDataset object
        An image dataset. 
    expected_len : int, optional
        The expected lenght of the dataset. The default is None.
    idx_to_get : int, optional
        The index to try and load. The default is 0.

    Returns
    -------
    None.

    '''
    # If given an expected length, check that the lenght is correct.
    if expected_len is not None:
        assert(len(dataset) == expected_len), 'Dataset size is not the expected size.'
    # Fetch item at index idx_to_get
    item = dataset.__getitem__(idx_to_get) 
    # If the dataset is empty, raise
    assert(len(dataset) > 0), 'Dataset is empty'
    # If the entry is not a dict, raise
    assert(type(item) == dict), 'The returned item should be a dictionary'
    # If the original image is not a tensor, raise
    assert(type(item['original_image']) == torch.Tensor), 'The original image should be a tensor'
    # If the edited image is not a tensor, raise
    assert(type(item['edited_image']) == torch.Tensor), 'The edited image should be a tensor'
    # If the dimensions of both images are not equal, raise
    assert(item['original_image'].shape == item['edited_image'].shape), 'The orginal image and the edited image should have the same dimension'
    
def perform_tests_dataloader(dataloader,plot = True):
    '''
    Checks the the dataloader can load data. Also can plot some example images.

    Parameters
    ----------
    dataloader : DataLoader object
        The DataLoader to be Tested
    plot : boolean, optional
        Indicates if example images should be ploted. The default is 'True'
        

    Returns
    -------
    None.

    '''
    # Gets the next output of the dataloader.
    out = next(iter(dataloader))
    # Size = number of datapoints in the batch.
    size = out['original_image'].shape[0]
    
    if plot:
        fig, axs = plt.subplots(size, 2,figsize = (5,10))
        # Iterate through each datapoint
        for i in range(size):
            # Get the original image
            original_image = out['original_image'][i,:,:,:]
            # Get the edited image
            edited_image = out['edited_image'][i,:,:,:]
            # Plot them side by side
            axs[i,0].imshow(original_image.permute(1,2,0))
            axs[i,1].imshow(edited_image.permute(1,2,0))
    
        # Label columns
        axs[0,0].set_title('original')
        axs[0,1].set_title('edited')
        
        # Format plot
        for ax in fig.get_axes():
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()


if __name__ == '__main__':
    img_size = [512, 512]
    # Create a dataset with all objects.
    n_files = 25000 # Expect 25,000 entries a priori.
    n_experts = 5 # Expect 5 experts a priori
    dataset = ImageDataset('D:\\fivek_dataset',transform = RandomCrop(img_size))
    perform_tests_dataset(dataset, 5000 * n_experts)
    dataloader = DataLoader(dataset, batch_size = 3, shuffle=True)
    perform_tests_dataloader(dataloader)
    
    # Get 80/20 test train splits. Form a series of True/False indicating if the value should be retained. 
    train_count = round((n_files/n_experts)* 0.8)
    test_count =  round((n_files/n_experts)* 0.2)
    split = pd.Series([0]*train_count +[1]*test_count).sample(frac = 1).reset_index(drop = True)

    


    # Form a train dataset and a train dataloader. Test both.
    train_dataset = ImageDataset('D:\\fivek_dataset',split = split, retain = 0, transform = RandomCrop(img_size))
    perform_tests_dataset(train_dataset,train_count * n_experts)
    train_dataloader = DataLoader(train_dataset, batch_size = 3, shuffle=True)
    perform_tests_dataloader(train_dataloader, plot = False)
    
    # Form a test dataset and a test dataloader. Test both.
    test_dataset = ImageDataset('D:\\fivek_dataset',split = split, retain = 1, transform = RandomCrop(img_size))
    perform_tests_dataset(test_dataset,test_count* n_experts)
    test_dataloader = DataLoader(test_dataset, batch_size = 3, shuffle=True)
    perform_tests_dataloader(test_dataloader, plot = False)
    
