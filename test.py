import os
import random
import pkgutil
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from data import *
from modules import *
from modules.acc import *
from models.model import MyModel
from peft import LoraConfig, get_peft_model,PeftModel

use_wandb = False
if pkgutil.find_loader("wandb") is not None:
    import wandb
    use_wandb = True

def test():
    args = parse_arguments()
    args.data_phase = 'train'
    if use_wandb: wandb_init(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyModel(args).to(device)
    src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=args.max_source_length)
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=args.max_target_length, use_fast=True, extra_ids=0, additional_special_tokens =[f"<loc_{i}>" for i in range(args.loc_vocab_size)])
    model.transformer.resize_token_embeddings(len(tgt_tokenizer))
    # checkpoints_names = [file_name for file_name in os.listdir(args.result_dir) if file_name.endswith('.pth') and file_name.startswith('epoch_')]
    # logger = get_logger(args, f'test_{args.datasets[0]}.log')
    dataset = get_dataset(args, dataset_name=args.datasets[0], phase=args.data_phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
    # for checkpoints_name in checkpoints_names:
    #     epoch = checkpoints_name.split('_')[1].split('.')[0]
    #     print(f'loading {checkpoints_name}...', end='')
    #     model.load(result_name=checkpoints_name)
    #     print(f'done')

    if args.loc_learn == "train":
        model.load(result_name=args.pth_path)
    elif args.loc_learn == "lora":
        model = PeftModel.from_pretrained(model, args.pth_path)


    dataloader = get_dataloader(args, dataset, num_workers=4, shuffle=False)
    random.seed(999)
    torch.manual_seed(999)
    srcs = []
    preds = []
    gts = []
    str_preds = []
    str_gts = []
    for src_images, tgt_images, src_texts, tgt_texts in tqdm(dataloader):
        with torch.no_grad():
            tgt_text = tgt_texts
            src_images = src_images.to(device)
            src_inputs = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt') # ['pt', 'tf', 'np', 'jax']
            src_texts = src_inputs['input_ids'].to(device, non_blocking=True)
            src_texts = src_texts.to(device, non_blocking=True)
            tgt_inputs = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')
            tgt_texts = tgt_inputs['input_ids'].to(device, non_blocking=True)
            tgt_texts = tgt_texts.to(device, non_blocking=True)
            src_attention_masks = torch.ones_like(src_texts, device=device)
            src_attention_masks[src_texts==0] = 0

            outputs = model(src_images, src_texts, src_attention_masks, return_loss=False)
            preds = tgt_tokenizer.batch_decode(outputs[:,1:-1])

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
            preds.extend(cleaned_preds)

    # print(gts)
    # print(preds)
    result, results = evaluate_score(gts, preds)
    print(result)
    print(results)
    if args.data_phase != 'train':
        preds = tgt_tokenizer.batch_decode(preds)
        gts = tgt_tokenizer.batch_decode(gts)
    # result['epoch'] = int(epoch)
    if use_wandb:
        if args.data_phase == 'train':
            my_table = wandb.Table(columns=["id", "Ground Truth", "Prediction", "Ground Truth (num)", "Prediction (num)"]+list(results.keys()))
            for id, contents in enumerate(zip(gts, preds, str_gts, str_preds, *results.values())):
                my_table.add_data(id+1, *contents)
            wandb.log({"results_ep"+str(args.num_epochs): my_table})
        wandb.log(result)
    wandb.finish()

def wandb_init(args):
    wandb.init(
        project=f"pretrain_test", 
        name=f"{args.datasets[0]}_{args.data_phase}_b{args.batch_size}",
        config=args,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("Bleu_*", step_metric="epoch")
    wandb.define_metric("CIDEr", step_metric="epoch")
    wandb.define_metric("ROUGE_L", step_metric="epoch")
    wandb.define_metric("results", step_metric="epoch")

if __name__ == '__main__':
    test()