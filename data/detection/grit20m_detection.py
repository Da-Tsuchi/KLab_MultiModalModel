import os
# from .pretrain import PretrainDatasetLoader
from ..dataset_loader import DatasetLoader
import torch
from torchvision import transforms
from PIL import Image


class GRIT20MDatasetLoader(DatasetLoader):
    def __init__(self, data_dir='/data/datatset/grit20m', phase='train'):
        super().__init__()
        if phase == 'train':
            tsv_path = os.path.join(data_dir, 'train_ref_exp.tsv')
        elif phase == 'val':
            tsv_path = os.path.join(data_dir, 'val_ref_exp.tsv')
        else:
            raise ValueError(f'Invalid phase: {phase}')

        with open(tsv_path, 'r') as f:
            lines = f.readlines()

        self.images, self.tgt_texts, self.src_texts = [], [], []
        for line in lines[1:10]:
            img_name, ref_exp , location = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir,img_name)
            # print(img_path)
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.src_texts.append(ref_exp)
                self.tgt_texts.append(location)
                