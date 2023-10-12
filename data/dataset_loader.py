import torch
from torchvision import transforms
from PIL import Image,PngImagePlugin

# Decompressed Data Too Largeになることを防ぐ
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, resize=256):
        self.images, self.tgt_texts, self.src_texts = [], [], []
        self.src_transforms = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.tgt_transforms = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        image, src_text, tgt_text = self.images[idx], self.src_texts[idx], self.tgt_texts[idx]
        image = Image.open(image).convert('RGB')
        src_image = self.src_transforms(image)
        tgt_image = self.src_transforms(image)

        return src_image, tgt_image, src_text, tgt_text
    
    def __len__(self):
        return len(self.images)
