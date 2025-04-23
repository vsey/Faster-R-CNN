import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
import torchvision

from lightning import LightningDataModule

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv

from typing import Tuple, Dict, Any, Union, Callable

from IPython.display import display
from PIL import Image



Tensor = torch.Tensor
T = torch.Tensor


with open('./pascal_label_map.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    LABEL_MAP = {key.strip("'"): int(value) for key, value in csv_reader}

print(LABEL_MAP)



class DataModule(LightningDataModule):
    def __init__(self, min_size = 800, max_size = 1333) -> None:
        super().__init__()
        # save hparams
        self.save_hyperparameters()


    # runs only once when DataModule gets created
    def prep_data(self) -> None:
        VOCDetection('./Datasets/VOC', download=True, image_set='train')
        VOCDetection('./Datasets/VOC', download=True, image_set='val')

    # runs on every node
    def setup(self, stage: str):

        self.train_dataset = VOCDetection('./Datasets/VOC', download=False, image_set='train', transforms=self.transform)
        self.val_dataset = VOCDetection('./Datasets/VOC', download=False, image_set='val', transforms=self.transform)


    def transform(self, image: Image.Image, targets: Dict[str, Dict[str, Any]]):

        # get the image dims
        image_width, image_height = image.size

        # find the short and long side of image
        short_side: int = min(image_width, image_height)
        long_side: int = max(image_width, image_height)

        # find the scale fators to make long side to max_size and short side to min_size
        short_scale_factor: float = float(self.hparams.min_size / short_side)
        long_scale_factor: float = float(self.hparams.max_size / long_side)

        # choose the smaller of the two ie least transformation (this preserves aspectratio)
        scale_factor: float = min(short_scale_factor, long_scale_factor)

        # Find new image size
        new_width: int = int(scale_factor * image_width)
        new_height: int =  int(scale_factor * image_height)
        
        # apply transform to image
        image = torchvision.transforms.functional.to_tensor(image)
        image = torchvision.transforms.functional.resize(image, (new_height, new_width))
        image = torchvision.transforms.functional.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # Target transform
        # get objects from target dict
        objects = targets['annotation']['object']

        # ectart labesl and bboxes
        processed_objects = [self.process_object(object) for object in objects]

        # convert boxes and labes to tensros
        labels = torch.stack([object['label'] for object in processed_objects])
        bboxes = torch.vstack([object['bbox'] for object in processed_objects])

        # scale bboxes to new shape
        scaled_bboxes = bboxes * scale_factor

        
        return image, {'labels': labels, 'bboxes': scaled_bboxes}


    @staticmethod
    def process_object(object):
        
        # extract label
        name = object['name']
        label = torch.tensor(LABEL_MAP[name], dtype=torch.int)

        # extarct bbox parameters
        bbox = object['bndbox']
        x_min = float(bbox['xmin'])
        y_min = float(bbox['ymin'])
        x_max = float(bbox['xmax'])
        y_max = float(bbox['ymax'])

        # stack to  tensor
        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float)

        return {'label': label, 'bbox': bbox}


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, shuffle=True, num_workers=4, pin_memory=True)
    

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None, shuffle=True, num_workers=2, pin_memory=True)

        

if __name__ == '__main__':
    # model = RCNN()

    # model.prep_data()
    # dataloader = model.train_dataloader()

    # tt = model.target_transform

    dm = DataModule()
    # dm.prep_data()
    dm.setup('fit')
    dataloader = dm.train_dataloader()



    for x, y in dataloader:

        print(type(x))

        plt.imshow(x.permute(1, 2, 0))
        
        # print(y)


