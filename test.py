import os
import re
import random
import pkgutil
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from tqdm import tqdm
from typing import List

from data import *
from modules import *
from modules.acc import *
from modules.utils import *
from models.model import MyModel
from peft import LoraConfig, get_peft_model,PeftModel
import argparse
# from torchinfo import summary

use_wandb = False
if pkgutil.find_loader("wandb") is not None:
    import wandb
    use_wandb = True

def compute_iou(bbox1: list, bbox2: list, verbose: bool=False):
    x1,y1,x2,y2 = bbox1
    x1_,y1_,x2_,y2_ = bbox2
    
    x1_in = max(x1,x1_)
    y1_in = max(y1,y1_)
    x2_in = min(x2,x2_)
    y2_in = min(y2,y2_)

    intersection = compute_area(bbox=[x1_in,y1_in,x2_in,y2_in],invalid=0.0)
    area1 = compute_area(bbox1,invalid=0)
    area2 = compute_area(bbox2,invalid=0)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou 


def compute_area(bbox: list, invalid: float=None) -> float:
    x1,y1,x2,y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1) * (y2 - y1)

    return area


def assign_boxes(pred_boxes: list[list], gt_boxes: list[list]):
    n1 = len(pred_boxes)
    n2 = len(gt_boxes)
    cost = np.zeros([n1,n2])
    ious = np.zeros([n1,n2])
    for i,bbox1 in enumerate(pred_boxes):
        for j,bbox2 in enumerate(gt_boxes):
            iou = compute_iou(bbox1,bbox2)
            ious[i,j] = iou
            cost[i,j] = 1-iou

    # solve assignment
    pred_box_ids, gt_box_ids = linear_sum_assignment(cost)
    pair_ids = list(zip(pred_box_ids, gt_box_ids))

    # select assignments with iou > 0
    pair_ids = [(i,j) for i,j in pair_ids if ious[i,j] > 0]
    pairs = [(pred_boxes[i],gt_boxes[j]) for i,j in pair_ids]
    pair_ious = [ious[i,j] for i,j in pair_ids]

    return pairs, pair_ious, pair_ids


def loc_metric(pred_boxes: list[list], gt_boxes: list[list]) -> float:
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)
    if num_pred == 0 and num_gt == 0:
        return 1
    elif min(num_pred,num_gt) == 0 and max(num_pred,num_gt) > 0:
        return 0
        
    pairs, pair_ious, pair_ids = assign_boxes(pred_boxes,gt_boxes)
    num_detected = len(pairs)
    num_missed = num_gt - num_detected
    return np.sum(pair_ious) / (num_pred + num_missed)

def find_loc_index_combinations(s):  
    # 正規表現のパターンを更新: <loc_数字>の後に物体の名前も取得できるようにする
    pattern = r'<loc_(\d+)><loc_(\d+)>'
    
    # Find all matches in the given string  
    matches = re.findall(pattern, s)  
      
    valid_combinations = []  
      
    for match in matches:
        x_val, y_val = match

        valid_combinations.append((int(x_val), int(y_val)))
      
    return valid_combinations

def get_box_coords_from_index(P, ul_idx, lr_idx):  
    """  
    Given a grid of length P and the indices of the upper-left and lower-right corners of a bounding box,  
    returns the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2].  
      
    Args:  
    - P (int): the length of the grid  
    - ul_idx (int): the index of the grid cell that corresponds to the upper-left corner of the bounding box  
    - lr_idx (int): the index of the grid cell that corresponds to the lower-right corner of the bounding box  
      
    Returns:  
    - box_coords (np.array of shape (4,)): the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2]  
    """  
    # Compute the size of each cell in the grid  
    cell_size = 1.0 / P  
      
    # Compute the x and y indices of the upper-left and lower-right corners of the bounding box  
    ul_x = ul_idx % P  
    ul_y = ul_idx // P  
      
    lr_x = lr_idx % P  
    lr_y = lr_idx // P  
      
    # Compute the normalized coordinates of the bounding box  
    if ul_idx == lr_idx:  
        x1 = ul_x * cell_size  
        y1 = ul_y * cell_size  
        x2 = lr_x * cell_size + cell_size  
        y2 = lr_y * cell_size + cell_size  
    elif ul_x == lr_x or ul_y == lr_y:  
        x1 = ul_x * cell_size  
        y1 = ul_y * cell_size  
        x2 = lr_x * cell_size + cell_size  
        y2 = lr_y * cell_size + cell_size  
    else:  
        x1 = ul_x * cell_size + cell_size / 2  
        y1 = ul_y * cell_size + cell_size / 2  
        x2 = lr_x * cell_size + cell_size / 2  
        y2 = lr_y * cell_size + cell_size / 2  
      
    return np.array([x1, y1, x2, y2])

def decode_bbox_from_caption(caption, quantized_size=40, **kwargs):
    
    valid_combinations = find_loc_index_combinations(caption)
    # entity_names = list(valid_combinations)
    patch_index_coords = list(map(lambda pair: get_box_coords_from_index(quantized_size, pair[0], pair[1]), valid_combinations))
    collect_entity_location = []
    for patch_index_coord in patch_index_coords:
        collect_entity_location.append(patch_index_coord.tolist())
    
    return collect_entity_location

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    # Compute the intersection area
    interArea = max(0, min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])) * max(0, min(boxA[3], boxB[3]) - max(boxA[1], boxB[1]))
    if interArea == 0:
        # print("interArea: ", interArea)
        return 0.0
    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # print("iou: ", iou)
    return iou

args = argparse.Namespace(
    # Model setting
    image_model_name="microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
    image_model_train=False,
    language_model_name="google/flan-t5-small",
    transformer_model_name="google/flan-t5-base",
    ffn=True,
    phase = "train",
    transformer_d_model=768,
    transformer_d_ff=3072,
    # transformer_d_model=1024,
    # transformer_d_ff=4096,
    transformer_d_kv=64,
    transformer_num_heads=12,
    transformer_num_layers=2,
    transformer_num_decoder_layers=12,
    image_vocab_size=16384,
    loc_vocab_size=1600,
    vae_ckpt_path="checkpoints/vqgan.pt",
    max_source_length=256,
    max_target_length=256,
    # Train setting
    pretrain="train", 
    # Dir setting
    root_dir="/data01/",
    result_dir="results/",
    loss = "CrossEntropy",
    loc_learn = "lora",
    float_type = 'bfloat16',
    lora_r = 4,
    lora_alpha = 4,
    lora_dropout = 0.1,
    lora_bias = "none"
)

def inference():
    args = parse_arguments()
    args = argparse.Namespace(
        # Model setting
        image_model_name="microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
        image_model_train=False,
        language_model_name="google/flan-t5-base",
        transformer_model_name="google/flan-t5-base",
        ffn=True,
        phase = "train",
        transformer_d_model=768,
        transformer_d_ff=3072,
        # transformer_d_model=1024,
        # transformer_d_ff=4096,
        transformer_d_kv=64,
        transformer_num_heads=12,
        transformer_num_layers=2,
        transformer_num_decoder_layers=12,
        image_vocab_size=16384,
        loc_vocab_size=1600,
        vae_ckpt_path="checkpoints/vqgan.pt",
        max_source_length=512,
        max_target_length=512,
        # Train setting
        pretrain="train", 
        # Dir setting
        root_dir="/user/data/",
        result_dir="results/",
        loss = "CrossEntropy",
        loc_learn = "train",
        float_type = 'bfloat16',
        lora_r = 4,
        lora_alpha = 4,
        lora_dropout = 0.1,
        lora_bias = "none",
        datasets = "openimage_loc",
        batch_size=10
    )

    args.data_phase = 'val'
    # logger = get_logger(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyModel(args).to(device)
    src_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", model_max_length=args.max_source_length, use_fast=True)
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=args.max_target_length, use_fast=True, extra_ids=0, additional_special_tokens =[f"<loc_{i}>" for i in range(1600)])
    
    # logger = get_logger(args, f'test_{args.datasets[0]}.log')
    dataset = get_dataset(args, dataset_name="openimage_loc", phase=args.data_phase)
    # processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-base", cache_dir=args.cache_dir)

    # モデルのトークン埋め込み層のサイズを更新
    # model.transformer.resize_token_embeddings(len(tgt_tokenizer))

    # path = "/home/tsuchida/instruct_blip/results/instructblip/lora/openimage_loc/enc_dec/epoch_18"
    path = "/home/tsuchida/KLab_MultiModalModel/results/1201/lora/2e-4/openimage_loc/enc2_dec12/epoch_50"

    # model = PeftModel.from_pretrained(model, path)
    model.eval()
    print(f'done')
    
    dataloader = get_dataloader(args, dataset, num_workers=4, shuffle=False)
    random.seed(999)
    torch.manual_seed(999)
    gts = []
    prs = []

    total_precision = 0.0  # Total True Positives across all images
    total_recall = 0.0  # Total False Positives across all images
    total_f = 0.0  # Total False Negatives across all images
    num = 0
    acc_array = []
    iou_threshold = 0.5  # IoU threshold for a prediction to be considered correct
    for src_image, tgt_images, src_text, tgt_text in tqdm(dataloader):
        with torch.no_grad():
            tgt_text2 = tgt_text
            src_images = src_image.to(device, non_blocking=True)
            # if args.phase == 'pretrain':
            #    tgt_images = tgt_images.to(device)
            #    tgt_texts, _ = model.module.image_to_z(tgt_images)
            if args.phase == 'classify':
                src_inputs = src_tokenizer(src_text, padding="longest", max_length=args.max_source_length, return_tensors='pt') # ['pt', 'tf', 'np', 'jax']
                src_texts = src_inputs['input_ids'].to(device, non_blocking=True)
                tgt_texts = tgt_texts.to(device, non_blocking=True)
                tgt_attention_masks = None
            else:
                src_inputs = src_tokenizer(src_text, padding="longest", max_length=args.max_source_length, return_tensors='pt') # ['pt', 'tf', 'np', 'jax']
                src_texts = src_inputs['input_ids'].to(device, non_blocking=True)
                src_texts = src_texts.to(device, non_blocking=True)
                tgt_inputs = tgt_tokenizer(tgt_text, padding="longest", max_length=args.max_target_length, return_tensors='pt')
                tgt_texts = tgt_inputs['input_ids'].to(device, non_blocking=True)
                tgt_texts = tgt_texts.to(device, non_blocking=True)
                tgt_attention_masks = torch.ones_like(tgt_texts, device=device, dtype=torch.bool)
                tgt_attention_masks[tgt_texts == 0] = 0
            src_attention_masks = torch.ones_like(src_texts, device=device, dtype=torch.bool)
            src_attention_masks[src_texts == 0] = 0
            
            preds = model(src_images, src_texts, src_attention_masks, tgt_texts, tgt_attention_masks,return_loss = False)
            
            # 予測の取得
            preds = tgt_tokenizer.batch_decode(preds[:, 1:-1])
            cleaned_preds = []
            prs = []
            for pred in preds:
                # '</s>'が現れる位置を見つけます
                end_pos = pred.find('</s>')
                # '</s>'が見つかった場合、その位置までの文字列を保持します
                if end_pos != -1:
                    pred = pred[:end_pos]

                # '<pad>'を削除します
                pred = pred.replace('<pad>', '')

                cleaned_preds.append(pred.strip())  # 空白を削除してリストに追加します
            prs.extend(cleaned_preds)

            # print('tgt_text:', tgt_text2)
            # print('pred:', prs)

            # バッチ状態から1つずつ予測と正解を取り出す
            for tgt_bbox, pred_bbox in zip(tgt_text2, prs):
                # 1枚の画像に対する正解値と予測値を座標のリストに変換
                tgt_bboxes = decode_bbox_from_caption(tgt_bbox)
                # print("tgt_bboxes: ", tgt_bboxes)
                pred_bboxes = decode_bbox_from_caption(pred_bbox)
                
                # print(float(loc_metric(pred_bboxes,tgt_bboxes)))
                acc_array.append(float(loc_metric(pred_bboxes,tgt_bboxes)))

                
                # # gt_match_status = [False] * len(tgt_bboxes)
                # # pred_match_status = [False] * len(pred_bboxes)
                # # 予測と正解のすべての組み合わせに対してIoUを計算
                # for pred_idx, pred_box in enumerate(pred_bboxes):
                #     # Iterate over each ground truth
                #     for gt_idx, gt_box in enumerate(tgt_bboxes):
                        
                #         # print("gt_box: ", gt_box)
                #         # print("pred_box: ", pred_box)
                #         if not gt_match_status[gt_idx]:  # If this ground truth is not matched yet
                #             iou = calculate_iou(gt_box, pred_box)
                #             if iou >= iou_threshold:
                #                 # If IoU is above the threshold, it's a match
                #                 gt_match_status[gt_idx] = True
                #                 pred_match_status[pred_idx] = True
                #                 break  # No need to check other ground truths for this prediction

                # # Update TP, FP, FN based on matches found
                # tps = sum(pred_match_status)
                # fps = len(pred_bboxes) - tps
                # fns = len(tgt_bboxes) - tps
                # print("tps: ", tps)
                # print("fps: ", fps)
                # print("fns: ", fns)

                # # 1枚につき適合率・再現率・F値を計算
                # precision = tps / (tps + fps) if (tps + fps) > 0 else 0
                # recall = tps / (tps + fns) if (tps + fns) > 0 else 0
                # f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  

                # total_precision += precision
                # total_recall += recall
                # total_f += f
                num += 1
    
    print("localize_acc:",np.mean(acc_array))
    # average_precision = total_precision / num
    # average_recall = total_recall / num
    # average_f = total_f / num

    # print(f"average_precision: {average_precision}")
    # print(f"average_recall: {average_recall}")
    # print(f"average_f: {average_f}")

if __name__ == '__main__':
    inference()
