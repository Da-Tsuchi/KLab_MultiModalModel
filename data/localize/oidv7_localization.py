import os
from ..dataset_loader import DatasetLoader

#存在しない画像を除外するためのリスト
dropimageidlist = ["7f1934f5884fad79","429019e83c1c2c94","4f818c006da84c9e","5b86e93f8654118a","673d74b7d39741c3","6dcd3ce37a17f2be","805baf9650a12710"
                   ,"98ac2996fc46b56d","a46a248a39f2d97c","000101475b6bc944"]

class OpenImageDataset_Localization(DatasetLoader):
    """openimageのdetectionデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/openimage/",phase:str="train"):
        super().__init__()        
        if phase=="val":
            phase = "validation"

        with open(os.path.join(data_dir,"tsv",f"{phase}_loc_cut_max_tokens.tsv")) as f:
            items = f.read()

        items = items.split("\n")
        items = [item.split("\t") for item in items]
        items = items[1:]
        self.tgt_texts = [item[3] for item in items]
        self.src_texts = [f"Which regions does the text \"{item[1]}\" describe?" for item in items]
        self.images = [os.path.join(data_dir,f"{phase}_256_png",f"{item[0]}.png") for item in items]

        #dropimageidlistに含まれる画像と対応するテキストを除外する
        drop_intdexs = []
        for drop_id in dropimageidlist:
            drop_path = os.path.join(data_dir,f"{phase}_256_png",f"{drop_id}.png")
            if drop_path in self.images:
                drop_intdexs.extend([i for i,x in enumerate(self.images) if x == drop_path])
        self.tgt_texts = [x for i,x in enumerate(self.tgt_texts) if i not in drop_intdexs]
        self.src_texts = [x for i,x in enumerate(self.src_texts) if i not in drop_intdexs]
        self.images = [x for i,x in enumerate(self.images) if i not in drop_intdexs]

    