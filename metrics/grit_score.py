import os
import re
import pkgutil
import numpy as np
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List
# from torchinfo import summary

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
