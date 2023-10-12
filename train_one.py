import wandb
import os
import random
import torch
import numpy as np
from transformers import AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt

from data import *
from modules import *
from models.model_decode import MyModel
from modules.acc import *
from modules.utils import *

def train():
    args = parse_arguments()

    os.makedirs(args.result_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger = get_logger(args)

    wandb.init(
    # set the wandb project where this run will be logged
        project="sample-project",
        # track hyperparameters and run metadata
        config=args,
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model
    model = MyModel(args).to(device)
    
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(args, optimizer)

    src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True)
    # tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True, extra_ids=0, additional_special_tokens =[f"<extra_id_{i}>" for i in range(100)] + [f"<loc_{i}>" for i in range(args.loc_vocab_size)] + [f"<img_{i}>" for i in range(args.image_vocab_size)])

    # データの設定
    train_dataset, val_dataset = get_data(args)
    train_loader = get_dataloader(args, train_dataset, num_workers=4, shuffle=False)

    if args.num_epochs is None:
        print("This code only supports num_epochs mode.")
        exit()

    steps = 0
    loss_counter = LossCounter()

    src_images, tgt_images, src_texts, tgt_texts = train_dataset[0]
    src_images = src_images.unsqueeze(0)
    tgt_images = tgt_images.unsqueeze(0)

    data_iter = iter(train_loader)
    src_images,tgt_images,src_texts,tgt_texts = data_iter.__next__()
    
    src_images = src_images.to(device)

    # if args.pretrain:
    #     tgt_images = tgt_images.to(device)
    #     tgt_texts, _ = model.image_to_z(tgt_images)

    # matches = [re.findall(pattern, tgt_text)[:256] for tgt_text in tgt_texts]
    # targets = [[int(m) for m in match] for match in matches]
    # targets = torch.tensor(targets).to(device)
    # targets = model.vae.decode_code(targets)
    # custom_to_pil(targets[0]).save(os.path.join(args.result_dir, "target.png"))

    # print("src_images.shape", src_images.shape)
    # print("tgt_images.shape", tgt_images.shape)
    print("src_texts", src_texts)
    print("tgt_texts", tgt_texts)
    src_texts = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(device) # ['pt', 'tf', 'np', 'jax']
    tgt_texts = src_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(device) # ['pt', 'tf', 'np', 'jax']
    print("src_texts", src_texts)
    print("tgt_texts", tgt_texts)
    train_count = 0
    for epoch in range(1, args.num_epochs+1):
        # 学習ループ
        tgt_text = tgt_texts
        
        
        train_count = torch.tensor(0).to(device)
        train_cider = torch.tensor(0.0).to(device)  # 正解のカウント用の変数を初期化
        train_bleu = torch.tensor(0.0).to(device)
        if args.image_model_train:
            model.image_model.train()
        model.transformer.train()
        
        optimizer.zero_grad()

        image_mask_ratio = 0.0

        gts = []
        prs = []
        
        train_count += src_images.shape[0]
        loss, pred = model(src_images, src_texts, tgt_texts, image_mask_ratio=image_mask_ratio)
        print(pred)
        preds = src_tokenizer.batch_decode(pred[:, 1:-1])
        print(preds)
        loss.backward()
        
        # for param in model.transformer.parameters():
        #     print(param.grad)
        train_loss = loss.item()

        optimizer.step()
        # pbar.update(1)
        steps += 1

        loss_counter.add("train", train_loss)

        if args.lr_scheduler != '':
            scheduler.step()
            
        # 予測の取得
        
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
        train_cider += torch.tensor(cider, dtype=torch.float32).to(device)
        train_bleu += torch.tensor(bleu, dtype=torch.float32).to(device)
        
        train_cider_acc = train_cider.float() / train_count  # 正答率を計算
        # cider_counter.add("train", train_cider_acc.cpu().numpy().copy())  # 正答率をAccCounterに追加
        train_bleu_acc = train_bleu.float() / train_count  # 正答率を計算

        logger.info(f'[Epoch ({epoch}/{args.num_epochs})] Train loss : {train_loss}, Steps : {steps}, CIDEr : {train_cider_acc}, BLEU : {train_bleu_acc}')
        wandb.log({"train_loss": train_loss, "train_cider": train_cider_acc, "train_bleu": train_bleu_acc})

        if (epoch) % 50 == 0:
            with torch.no_grad():
                # outputs = model(src_images, src_texts, tgt_texts, return_loss=False, num_beams=4)
                outputs = model(src_images, src_texts, tgt_texts, return_loss=False)
                preds = src_tokenizer.batch_decode(outputs)

                print(f"Generated text: {preds}")
                # matches = []
                # for pred in preds:
                #     match = re.findall(pattern, pred)
                #     if len(match) >= 256:
                #         matches.append([int(m) for m in match[:256]])

                # if len(matches) > 0:
                #     preds = torch.tensor(matches).to(device)
                #     preds = model.vae.decode_code(preds)
                #     custom_to_pil(preds[0]).save(os.path.join(args.result_dir, f"epoch_{epoch}.png"))
                #     print(f"Generated image {epoch} saved")
    
        if args.save_interval is not None:
            if (epoch) % args.save_interval == 0:
                print(f'Model {epoch} saving...')
                model.save(result_name=f'epoch_{epoch}.pth')
                print(f'Model {epoch} saved')
            
    loss_counter.plot_loss(args.result_dir)

    # 結果をcsvに1行で保存
    split_result = args.result_dir.split("/")
    csv_path = os.path.join(split_result[0], split_result[1], split_result[2], f"{split_result[3]}.csv")

    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("image_model_name,language_model_name,transformer_num_layers,transformer_num_decoder_layers,seed,lr,optimizer,lr_scheduler,batch_size,num_epochs,datasets,train_loss,result_dir\n")
            
    wandb.alert(
        title='学習が完了しました。',
        text='学習が完了しました。結果を確認してください。',
    )

    wandb.finish()

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

if __name__=="__main__":
    train()