import sys
sys.path.append("/home/tsuchida/KLab_MultiModalModel/tsuchida_workdir/..")

import os
import random
import torch
import numpy as np
from transformers import AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt

from modules.acc import *
from modules.utils import *

gts = []
prs = []
        
preds = ["This is a pen.","I am a doctor.","I am a doctor."]
tgt_text = ["He is a pen.","I am an doctor.","I am a doctor."]

cleaned_preds = []
for pred in preds:
    # '</s>'が現れる位置を見つけます
    end_pos = pred.find('</s>')
    # '</s>'が見つかった場合、その位置までの文字列を保持します
    if end_pos != -1:
        pred = pred[:end_pos]

    # '<pad>'を削除します
    pred = pred.replace('<pad>', '')

    cleaned_preds.append(pred.strip())  # 空白を削除してリストに追加します
            # print(cleaned_preds)
            # print(tgt_text)
gts.extend(tgt_text)
prs.extend(cleaned_preds)

# 予測と実際のテキストが一致するかどうかを確認
cider,bleu = evaluate_score(prs, gts)
print("cider:",cider)
print("Bleu:",bleu)