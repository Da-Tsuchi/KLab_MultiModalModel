import os
import tarfile
from PIL import Image
from .pretrain import ClassifyPretrainDatasetLoader

class ImageNet21kPretrainDatasetLoader(ClassifyPretrainDatasetLoader):
    def __init__(self, args, data_dir='/data01/imagenet_21k/', phase="train",resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)

        # # Load class names
        # ids_txt_path = os.path.join(data_dir, 'imagenet21k_wordnet_ids.txt')
        # class_name_txt_path = os.path.join(data_dir, 'imagenet21k_wordnet_lemmas.txt')

        # with open(ids_txt_path, 'r') as f:
        #     ids = [line.strip() for line in f.readlines()]

        # with open(class_name_txt_path, 'r') as f:
        #     class_names = [line.strip() for line in f.readlines()]

        # imagenet_classes = dict(zip(ids, class_names))

        # img_folder_path = os.path.join(data_dir, 'images_256/')
        # folders = os.listdir(img_folder_path)
        # for folder in folders:
        #     folder_path = os.path.join(img_folder_path, folder)
        #     if os.path.isdir(folder_path):
        #         for img_name in os.listdir(folder_path):
        #             img_path = os.path.join(folder_path, img_name)
        #             self.images.append(img_path)
        #             self.src_texts.append(imagenet_classes[folder])

        # modeに基づいてtsvファイルの名前を決定
        tsv_filename1 = f"img_{phase}_256fix.tsv"
        tsv_path1 = os.path.join(data_dir, tsv_filename1)

        tsv_filename2 = f"text_{phase}_256fix.tsv"
        tsv_path2 = os.path.join(data_dir, tsv_filename2)
        
        # 画像のパスとラベルのマッピングを辞書に保存
        img_label_mapping = {}
        
        with open(tsv_path1, 'r') as f:
            for line in f:
                img_path_relative, label = line.strip().split('\t')
                img_path = os.path.join(data_dir, img_path_relative)
                img_label_mapping[img_path] = label

        with open(tsv_path2, 'r') as f:
            for line in f:
                img_path_relative, label = line.strip().split('\t')
                img_path = os.path.join(data_dir, img_path_relative)
                img_label_mapping[img_path] = label

        img_folder_path = os.path.join(data_dir, 'images_256/')
        folders = os.listdir(img_folder_path)
        for folder in folders:
            folder_path = os.path.join(img_folder_path, folder)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    if img_path in img_label_mapping:
                        self.images.append(img_path)
                        self.src_texts.append('What does the image describe ?')
                        self.tgt_texts.append(img_label_mapping[img_path])
                        # self.tgt_texts.append(f'a photo of {img_label_mapping[img_path]}')
                        # self.images.append(img_path)
        #             self.src_texts.append(imagenet_classes[folder])
                        # print(img_path, img_label_mapping[img_path])