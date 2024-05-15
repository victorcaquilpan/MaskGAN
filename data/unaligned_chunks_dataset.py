import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
from torchvision import io
from monai.transforms import Rand2DElasticd
import pandas as pd
import torch
import os

class UnalignedChunksDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + f'_{opt.Aclass}')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + f'_{opt.Bclass}')  # create a path '/path/to/data/trainB'
        self.dir_maskA = os.path.join(opt.dataroot, opt.phase + f'_mask{opt.Aclass}')  # create a path '/path/to/data/trainA'
        self.dir_maskB = os.path.join(opt.dataroot, opt.phase + f'_mask{opt.Bclass}')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.maskA_paths = sorted(make_dataset(self.dir_maskA, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.maskB_paths = sorted(make_dataset(self.dir_maskB, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        # Get the size of dataset A
        self.A_size = len(self.A_paths)  
        # Get the size of dataset B
        self.B_size = len(self.B_paths)  
        btoA = opt.direction == 'BtoA'
        self.half = opt.half
        if opt.half:
            self.A_size, self.B_size = self.A_size//2, self.B_size//2
            self.A_paths, self.B_paths = self.A_paths[:self.A_size], self.B_paths[self.B_size:]
        
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=True)
        self.transform_B = get_transform(self.opt, grayscale=True)

        # Check how many 3D images we are considering
        A_paths_main_img = [img.split("/")[-1] for img in self.A_paths]
        A_paths_main_img = [img.split("_")[0] for img in A_paths_main_img]
        self.number_3d_images_A = len(np.unique(A_paths_main_img))
        # Same for set B
        B_paths_main_img = [img.split("/")[-1] for img in self.B_paths]
        B_paths_main_img = [img.split("_")[0] for img in B_paths_main_img]        
        self.number_3d_images_B = len(np.unique(B_paths_main_img))

        # Save relative position of each img
        self.relative_pos_A = [int(img.split(".")[-2].split("_")[-1]) for img in self.A_paths]
        self.relative_pos_B = [int(img.split(".")[-2].split("_")[-1]) for img in self.B_paths]

        # Extract the name of images
        self.image_names_A = [name.split('/')[-1].replace('.png','') for name in self.A_paths]
        self.image_names_B = [name.split('/')[-1].replace('.png','') for name in self.B_paths]

        # Extract the name of the base image
        self.base_names_A = [img.split("_")[1] if 'paired' in img else img.split("_")[0] for img in self.image_names_A]
        self.base_names_B = [img.split("_")[1] if 'paired' in img else img.split("_")[0] for img in self.image_names_B]

        # Get the unique values
        self.vol_imgsA = np.unique(self.base_names_A)
        self.vol_imgsB = np.unique(self.base_names_B)

        # Get the size of volumetric images
        self.vol_sizeA = len(self.vol_imgsA)
        self.vol_sizeB = len(self.vol_imgsB)
        
        # Set a margin to use images from a similar relative position
        self.phase = opt.phase
        if self.phase == 'train':
            self.position_based_range = opt.position_based_range # This is a percentage
            # Define the age range
            self.range_months = opt.range_months

        
        if self.opt.phase != 'test':
            # Loading the CSV file
            self.feature_images = pd.read_csv(opt.feature_images_file_path)
            # Remove duplicate rows
            self.feature_images = self.feature_images.drop_duplicates()
            # Getting the age of each CT and MRI
            self.ages_images_A = []
            self.ages_images_B = []
            for img in self.vol_imgsA:
                age = self.feature_images.loc[(self.feature_images['new_name'] == float(img)) & (self.feature_images['Modality'] == "MR"), "PatientAgeMonths"].values
                if age.size > 0:
                    self.ages_images_A.append(age[0])
            for img in self.vol_imgsB:
                age = self.feature_images.loc[(self.feature_images['new_name'] == float(img)) & (self.feature_images['Modality'] == "CT"), "PatientAgeMonths"].values
                if age.size > 0:
                    self.ages_images_B.append(age[0])

        # Define a number of consecutive number of images to process
        self.number_slices = opt.n_slices
        self.serial_batches = opt.serial_batches

        # Check number of chunks
        self.chunks_per_imgA = [self.base_names_A.count(img) for img in self.vol_imgsA]
        self.chunks_per_imgB = [self.base_names_B.count(img) for img in self.vol_imgsB]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if index == 0:
            index_A = 0
        else: 
            index_A = (index * self.number_slices)

        # Loading A imgs and A masks
        A_paths = self.A_paths[index_A:index_A + self.number_slices] 
        maskA_paths = self.maskA_paths[index_A:index_A + self.number_slices]

        if self.phase == 'train' and not self.serial_batches:
            # Get slices of A
            A_slices = self.image_names_A[index_A:index_A + self.number_slices]
            A_vol_img = A_slices[0][0:3]
            A_months = self.ages_images_A[list(self.vol_imgsA).index(A_vol_img)]
            # Select images which are in a similar age
            potential_indexes_months = [index for index, value in enumerate(self.ages_images_B) if (A_months-self.range_months) <= value <= (A_months + self.range_months)]
            # Define  volumetric B image
            index_months = random.randint(0, len(potential_indexes_months) - 1)
            index_B = potential_indexes_months[index_months]
            # Selecting the B image
            B_vol_img = self.vol_imgsB[index_B]
            # Select the slices of that volume
            B_slices = [slice for slice in self.image_names_B if B_vol_img in slice[0:3]]
            # Extract the absolute position of A
            absolute_pos_A = os.path.basename(A_paths[0]).split("_")[1]
            # Select the specific B slices
            #B_slices = B_slices[int(absolute_pos_A): int(absolute_pos_A) + self.number_slices]
            B_initial_slice = [slice for slice in B_slices if absolute_pos_A == slice[4:7]][0]
            B_paths = [self.B_paths[slice] for slice in range(self.image_names_B.index(B_initial_slice),self.image_names_B.index(B_initial_slice) + self.number_slices)]                
            # Get the B masks    
            maskB_paths = [self.maskB_paths[slice] for slice in range(self.image_names_B.index(B_initial_slice),self.image_names_B.index(B_initial_slice) + self.number_slices)]

        elif self.phase == 'val' or self.phase == 'test' or self.serial_batches:
            B_paths = self.B_paths[index_A:index_A + self.number_slices]
            maskB_paths = self.maskB_paths[index_A:index_A + self.number_slices]

        # Load images and mask
        for idx in range(0,self.number_slices):

            A_path = A_paths[idx]
            B_path = B_paths[idx]
            maskA_path = maskA_paths[idx]
            maskB_path = maskB_paths[idx]

            A_img = io.read_image(A_path)
            B_img = io.read_image(B_path)
        
            A_mask = io.read_image(maskA_path)
            B_mask = io.read_image(maskB_path)

            # apply image transformation to standarize data
            A, A_mask = self.transform_A(A_img, A_mask)
            B, B_mask = self.transform_B(B_img, B_mask)

            if idx == 0:
                A_chunk = A
                B_chunk = B
                A_mask_chunk = A_mask
                B_mask_chunk = B_mask
                A_paths_chunks = [A_path]
                B_paths_chunks = [B_path]
            else:
                A_chunk = torch.cat((A_chunk, A), dim = 0)         
                B_chunk = torch.cat((B_chunk, B), dim = 0)
                A_mask_chunk = torch.cat((A_mask_chunk, A_mask), dim = 0)         
                B_mask_chunk = torch.cat((B_mask_chunk, B_mask), dim = 0)
                A_paths_chunks = A_paths_chunks + [A_path]
                B_paths_chunks = B_paths_chunks + [B_path]

        return {'A': A_chunk, 'B': B_chunk, 'A_paths': A_paths_chunks, 'B_paths': B_paths_chunks,
        'A_mask': A_mask_chunk, 'B_mask': B_mask_chunk}

    def __len__(self):
        """Return the total number of chunk images in the dataset.
        """
        if self.half:
            return (int(self.vol_sizeB*(self.chunks_per_imgB[0]/self.number_slices))) //2
        else:
            return int(self.vol_sizeB*(self.chunks_per_imgB[0]/self.number_slices)) 



