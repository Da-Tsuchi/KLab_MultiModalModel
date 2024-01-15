import json
import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class Vqa2dataset(DatasetLoader):
    def __init__(self,data_dir="/data/dataset/vqa2",phase="train", **kwargs):
        super().__init__(**kwargs)
        question_path = os.path.join(data_dir, f"v2_OpenEnded_mscoco_{phase}2014_questions.json")
        answer_path = os.path.join(data_dir, f"v2_mscoco_{phase}2014_annotations.json")
        quetions = json.load(open(question_path))
        answers = json.load(open(answer_path))
        count = 0

        for question, answer in zip(quetions['questions'], answers['annotations']):
            if count >= MAX_VAL_DATA_SIZE and phase == "val":
                break
            self.src_texts.append(question["question"])
            self.tgt_texts.append(answer["multiple_choice_answer"])
            self.images.append(f'{data_dir}/{phase}2014_256_png/COCO_{phase}2014_{str(question["image_id"]).zfill(12)}.png')
            count += 1
    # print(data)