import os
from .pretrain import ClassifyPretrainDatasetLoader

class ImageNetPretrainDatasetLoader(ClassifyPretrainDatasetLoader):
    def __init__(self, args, data_dir='/data/datatset/imagenet_2012', phase='train', resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)
        img_folder_path = os.path.join(data_dir, phase)

        # Load class names
        map_clsloc_path = os.path.join(data_dir, 'map_clsloc.txt')
        imagenet_classes = {}
        with open(map_clsloc_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                imagenet_classes[line[0]] = line[2]

        for class_folder, class_name in imagenet_classes.items():
            class_folder_path = os.path.join(img_folder_path, class_folder)
            class_name = class_name.replace('_', ' ')
            for img in os.listdir(class_folder_path):
                img_path = os.path.join(class_folder_path, img)
                self.images.append(img_path)
                self.src_texts.append('What does the image describe ?')
                self.tgt_texts.append(f"{class_name}")
                # self.src_texts.append(f'{class_name}.')
