import os
import re
from copy import deepcopy

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from ..dataset_loader import DatasetLoader

class VisualGenomeDetectDataset(DatasetLoader):
    """openimageのdetectionデータセット
    """    
    def __init__(self,data_dir:str="/data01/visual_genome/",phase:str="train"):
        super().__init__()        
        if phase=="val":
            phase = "val"

        with open(os.path.join(data_dir,f"{phase}_detect_fix.tsv")) as f:
            items = f.read()

        
        items = items.split("\n")
        items = [item.split("\t") for item in items]
        items = items[1:]
        # items = [item for item in items]
        items = [item for item in items if len(item)==2]
        self.src_texts = ["What objects are in the image?"]*len(items)
        self.tgt_texts = [item[1] for item in items]
        self.images = [os.path.join(data_dir,f"images_256",f"{item[0]}.png") for item in items]