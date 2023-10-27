import sys
sys.path.append("/home/tsuchida/KLab_MultiModalModel/tsuchida_workdir/..")

from PIL import Image
import torch
from models.model import MyModel
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from tqdm import tqdm

from data import *
from modules import *
from models.model import MyModel
from modules.acc import AccCounter

import argparse
args = argparse.Namespace(
    # Model setting
    image_model_name="microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
    image_model_train=False,
    language_model_name="google/flan-t5-small",
    ffn=True,
    transformer_d_model=768,
    transformer_d_ff=3072,
    transformer_d_kv=64,
    transformer_num_heads=12,
    transformer_num_layers=2,
    transformer_num_decoder_layers=1,
    image_vocab_size=16384,
    loc_vocab_size=1000,
    vae_ckpt_path="checkpoints/vqgan.pt",
    max_source_length=256,
    max_target_length=256,
    batch_size = 170,
    num_epochs = 100,
    loss = "FocalLoss",
    # Train setting
    pretrain=True, 
    datasets="imagenet",
    # Dir setting
    root_dir="/user/data/",
    result_dir="/home/tsuchida/KLab_MultiModalModel/results/pretrain_claa/imagenet/enc2_dec1/",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

dist.init_process_group("nccl")
rank = dist.get_rank()

# args = parse_arguments()
# args.gpu_nums = torch.cuda.device_count() # GPU数
# device_id = rank % args.gpu_nums
device_id =0

model = MyModel(args).to(device)
# print(model)
# 読み込みたい重みのパスを指定
path = "best.pth"
model.load(result_name=path)
model.eval()

from transformers import AutoTokenizer
from torchvision import transforms
src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256)
tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True, extra_ids=0, additional_special_tokens =[f"<extra_id_{i}>" for i in range(100)] + [f"<loc_{i}>" for i in range(1000)] + [f"<img_{i}>" for i in range(args.image_vocab_size)])
resize=256
src_transforms = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
])
tgt_transforms = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
])

from data import get_dataset
# train_dataset = get_dataset(args, dataset_name='imagenet', phase='train', src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
val_dataset = get_dataset(args, dataset_name='imagenet', phase='val', src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)

print("start")
count = 0
num = 0

# train_loader = get_distributed_dataloader(args, train_dataset, shuffle=True)
val_loader = get_distributed_dataloader(args, val_dataset, shuffle=False)
correct_count = torch.tensor(0).to(device_id)  # 正解のカウント用の変数を初期化
val_loop = tqdm(val_loader, desc=f'Val (Epoch {7}/{args.num_epochs})',disable=(rank != 0))
# __, val_dataset = get_data(args, src_tokenizer, tgt_tokenizer)
acc_counter = AccCounter()
val_count = torch.tensor(0).to(device)
correct_count = torch.tensor(0).to(device_id)  # 正解のカウント用の変数を初期化
for src_images, tgt_images, src_texts, tgt_texts in val_loop:
    with torch.no_grad():
        tgt_text = tgt_texts
        src_images = src_images.to(device_id, non_blocking=True)
        # if args.pretrain:
        #    tgt_images = tgt_images.to(device_id)
        #    tgt_texts, _ = model.module.image_to_z(tgt_images)
        src_texts = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(device_id, non_blocking=True) # ['pt', 'tf', 'np', 'jax']
        tgt_texts = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(device_id, non_blocking=True) # ['pt', 'tf', 'np', 'jax']
        
        outputs = model(src_images, src_texts, tgt_texts,return_loss=False)
        
        # val_loss += loss.item() * src_images.shape[0]
        val_count += src_images.shape[0]
        
        # 予測の取得
        
        preds = tgt_tokenizer.batch_decode(outputs[:, 1:-1])
        cleaned_preds = [pred.replace('</s>', '').replace('<pad>', '') for pred in preds]

        # 予測と実際のテキストが一致するかどうかを確認
        corrects = [1 if pred == actual else 0 for pred, actual in zip(cleaned_preds, tgt_text)]
        correct_count += sum(corrects)
        print("label:",tgt_text)
        print("pred:",cleaned_preds)

    # 他のノードから集める
    # dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_count, op=dist.ReduceOp.SUM)

    if rank == 0:
        # val_loss /= val_count
        # loss_counter.add("val", val_loss.cpu().numpy().copy())
        val_acc = correct_count.float() / val_count  # 正答率を計算
        acc_counter.add("val", val_acc.cpu().numpy().copy())  # 正答率をAccCounterに追加

        print(val_acc)
        print(val_count)
        print(correct_count)
    

# print("val")
# print(count)
# print(num)
# print(count/num)

# count = 0
# num = 0
# for src_image, tgt_image, src_text, tgt_text in train_dataset:
#     src_image = src_image.unsqueeze(0).to(device)
#     # print('src_text:', src_text)
#     # print('tgt_text:', tgt_text)
#     src_text = src_tokenizer(src_text, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(device) # ['pt', 'tf', 'np', 'jax']
#     tgt_text_de = tgt_tokenizer(tgt_text, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(device) # ['pt', 'tf', 'np', 'jax']
#     # print(src_text, tgt_text)

#     # display(custom_to_pil(src_image[0]))
#     output = model(src_image, src_text, tgt_text_de, return_loss=False, num_beams=4)
#     preds = tgt_tokenizer.batch_decode(output[:,1:-1])
#     num += 1
#     # print("tgt_text:",tgt_text)
#     # print("preds:",preds[0])
#     if(preds[0]==tgt_text):
#         count+=1

# print("train")
# print(count)
# print(num)
# print(count/num)