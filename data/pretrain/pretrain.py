import torch
import random
from PIL import Image
from torch.nn.functional import pad
from ..dataset_loader import DatasetLoader

class PretrainDatasetLoader(DatasetLoader):
    def __init__(self, args, resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(resize)
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length 
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.mask_tokens = src_tokenizer.additional_special_tokens_ids
        self.mask_prob = mask_probability

    def __getitem__(self, idx):
        image, text = self.images[idx], self.src_texts[idx]
        src_text = self.tgt_tokenizer.encode_plus(text, return_attention_mask=False, verbose=False)["input_ids"][:-1]
        tgt_text = self.generate_target_ids(src_text)
        src_text = self.tgt_tokenizer.decode(src_text)
        tgt_text = self.tgt_tokenizer.decode(tgt_text)

        image = Image.open(image).convert('RGB')#.resize((256,256))
        src_image = self.src_transforms(image)
        tgt_image = self.tgt_transforms(image)
        tgt_image = 2.*tgt_image-1.

        return src_image, tgt_image, src_text, tgt_text
    
    def generate_target_ids(self, input_id):
        target_id = []
        masked_indexes = sorted(random.sample(range(0, len(input_id)),  # sample a word index in sentence
                                                min(max(int(self.mask_prob * len(input_id)),1),  # number of tokens masked
                                                    len(self.mask_tokens) - 1)))  # but never more than special tokens available
        mask = [(i in masked_indexes)  # this is True or False
                for i in range(len(input_id))]
        i = 0
        end = len(input_id)
        masked_spans_counter = 0
        while i < end:
            if mask[i]:
                current_words_masked = [input_id[i]]
                input_id[i] = self.mask_tokens[masked_spans_counter]
                masked_spans_counter += 1
                while i + 1 < end and mask[i + 1]:
                    current_words_masked.append(input_id[i + 1])
                    del input_id[i + 1]
                    del mask[i + 1]
                    end -= 1
                target_id.extend(current_words_masked)
            else:
                if len(target_id) == 0 or target_id[-1] != self.mask_tokens[masked_spans_counter]:
                    target_id.append(self.mask_tokens[masked_spans_counter])
            i += 1
        return target_id
    
class ClassifyPretrainDatasetLoader(PretrainDatasetLoader):
    def __init__(self, args, resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)

    def __getitem__(self, idx):
        image, src_text,tgt_text = self.images[idx], self.src_texts[idx],self.tgt_texts[idx]
        rate = random.random()

        # if rate < 0.25:
        #     text = 'A short image description: ' + text
        # elif rate < 0.5:
        #     text = 'An image of ' + text
        # elif rate < 0.75:
        #     text = 'A photo of ' + text
        # else:
        #     text = 'An image that shows ' + text
        
        src_text = self.tgt_tokenizer.encode_plus(src_text, return_attention_mask=False, verbose=False)["input_ids"][:-1]
        tgt_text = self.tgt_tokenizer.encode_plus(tgt_text, return_attention_mask=False, verbose=False)["input_ids"][:-1]
        # tgt_text = self.generate_target_ids(src_text)
        src_text = self.tgt_tokenizer.decode(src_text)
        tgt_text = self.tgt_tokenizer.decode(tgt_text)

        image = Image.open(image).convert('RGB')#.resize((256,256))
        src_image = self.src_transforms(image)
        tgt_image = self.tgt_transforms(image)
        tgt_image = 2.*tgt_image-1.

        return src_image, tgt_image, src_text, tgt_text