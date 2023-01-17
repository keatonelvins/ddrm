import os
import numpy as np
import pandas as pd
from .vision import VisionDataset

class DiffuserDataset_preprocessed(VisionDataset):
    """Diffuser dataset https://waller-lab.github.io/LenslessLearning/dataset.html"""

    def __init__(self, csv_file, data_dir, label_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the Diffuser images.
            label_dir (string): Directory with all the natural images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_contents = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        
        
    def __len__(self):
        return len(self.csv_contents)

    def __getitem__(self, idx):

        img_name = self.csv_contents.iloc[idx,0]

        path_diffuser = os.path.join(self.data_dir, img_name) 
        path_gt = os.path.join(self.label_dir, img_name)
        
        image = np.load(path_diffuser+'.npy')
        label = np.load(path_gt+'.npy')

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample