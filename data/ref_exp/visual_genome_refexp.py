import os
import re
from copy import deepcopy

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from ..dataset_loader import DatasetLoader

class VisualGenomeRefExpDataset(DatasetLoader):
    """openimageのdetectionデータセット
    """    
    def __init__(self,data_dir:str="/data01/visual_genome/",phase:str="train"):
        super().__init__()        
        if phase=="val":
            phase = "val"

        with open(os.path.join(data_dir,f"{phase}_ref_exp.tsv")) as f:
            items = f.read()

        
        items = items.split("\n")
        items = [item.split("\t") for item in items]
        items = items[1:]
        # items = [item for item in items]
        items = [item for item in items if len(item)==3]
        self.src_texts = [f'Which regions does the text "{item[1]}" describe?' for item in items]
        self.tgt_texts = [item[2] for item in items]
        self.images = [os.path.join(data_dir,f"images_256",f"{item[0]}.png") for item in items]