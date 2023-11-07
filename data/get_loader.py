import os
import torch
from torch.utils.data import random_split, DataLoader, distributed, ConcatDataset
from .caption import *
from .caption.cc3m import CC3MDatasetLoader
from .image_classify import *
from .vqa import *
from .pretrain import *
from .relationship import *
from .detection import *
from .detection.oidv7_detection import OpenImageDataset
from .vqa.gqa import GQADataset
from .ref_exp.visual_genome_refexp import *
from .detection.vg_detection import *
from .localize.vg_localize import *
from .categorize.vg_categorize import *
from .relationship.vg_relationship import *

def get_data(args, src_tokenizer=None, tgt_tokenizer=None):
    train_datasets, val_datasets = [], []
    for dataset_name in args.datasets:
        if dataset_name in ['redcaps', 'sun397', 'cc12m']:
            dataset = get_dataset(args, dataset_name, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
            val_rate = 0.1
            val_size = int(len(dataset) * val_rate)
            train_size = len(dataset) - val_size

            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
            )
            
        else:
            train_dataset = get_dataset(args, dataset_name, phase="train", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
            val_dataset = get_dataset(args, dataset_name, phase="val", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    if len(args.datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)

    elif len(args.datasets) == 0:
        raise NotImplementedError
    
    return train_dataset, val_dataset

def get_distributed_dataloader(args, dataset, num_workers=4, shuffle=True):
    sampler = distributed.DistributedSampler(dataset, drop_last=True, shuffle=shuffle)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, sampler=sampler)
    return dataloader

def get_dataloader(args, dataset, num_workers=4, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=shuffle)
    return dataloader

def get_dataset(args, dataset_name, phase="train", src_tokenizer=None, tgt_tokenizer=None):
    data_dir = os.path.join(args.root_dir, dataset_name)
    if args.phase == "pretrain": # 事前学習だったら
        if src_tokenizer is None or tgt_tokenizer is None:
            raise NotImplementedError
        if 'redcaps' == dataset_name:
            dataset = RedCapsPretrainDatasetLoader(args, data_dir, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'imagenet' == dataset_name:
            dataset = ImageNetPretrainDatasetLoader(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'imagenet_21k' == dataset_name:
            dataset = ImageNet21kPretrainDatasetLoader(args, data_dir, phase=phase,src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'places365' == dataset_name:
            dataset = Places365PretrainDatasetLoader(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'sun397' == dataset_name:
            dataset = SUN397PretrainDatasetLoader(args, data_dir, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'inaturalist' == dataset_name:
            dataset = INaturalistPretrainDatasetLoader(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'cc3m' == dataset_name:
            dataset = CC3MPretrainDatasetLoader(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'cc12m' == dataset_name:
            dataset = CC12MPretrainDatasetLoader(args, data_dir, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        else:
            raise NotImplementedError
    else:
        if 'mscoco' == dataset_name:
            dataset = COCODatasetLoader(data_dir=os.path.join(args.root_dir+"mscoco2017"), phase =phase)
        elif 'redcaps' == dataset_name:
            dataset = RedCapsDatasetLoader(data_dir)
        elif 'cc3m' == dataset_name:
            dataset = CC3MDatasetLoader(data_dir,phase)
        elif 'vcr' == dataset_name:
            dataset = Vcrdataset(data_dir, phase=phase)
        elif 'vqa2' == dataset_name:
            dataset = Vqa2dataset(data_dir, phase=phase)
        elif 'imsitu' == dataset_name:
            dataset = imSituDataset(data_dir, phase=phase)
        elif 'imagenet' == dataset_name:
            dataset = ImageNetDatasetLoader(data_dir, phase=phase)
        # elif 'imagenet_21k' == dataset_name:
        #     dataset = ImageNet21kDatasetLoader(data_dir, phase=phase)
        elif 'gqa' == dataset_name:
            dataset = GQADataset(data_dir, phase=phase)

        elif 'openimage' == dataset_name:
            dataset = OpenImageDataset(data_dir, phase=phase)
        elif 'visual_genome_refexp' == dataset_name:
            dataset = VisualGenomeRefExpDataset(data_dir=os.path.join(args.root_dir+"visual_genome"), phase=phase)
        elif 'visual_genome_detect' == dataset_name:
            dataset = VisualGenomeDetectDataset(data_dir=os.path.join(args.root_dir+"visual_genome"), phase=phase)
        elif 'visual_genome_localize' == dataset_name:
            dataset = VisualGenomeLocalizeDataset(data_dir=os.path.join(args.root_dir+"visual_genome"), phase=phase)
        elif 'visual_genome_categorize' == dataset_name:
            dataset = VisualGenomeCategorizeDataset(data_dir=os.path.join(args.root_dir+"visual_genome"), phase=phase)
        elif 'visual_genome_relationship' == dataset_name:
            dataset = VisualGenomeRelationDataset(data_dir=os.path.join(args.root_dir+"visual_genome"), phase=phase)
        else:
            raise NotImplementedError
    return dataset