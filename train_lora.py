import os
import random
import numpy as np
import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

from data import *
from modules import *
from models.model_lora import MyModel
from modules.acc import AccCounter
from modules.utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    args = parse_arguments()
    args.gpu_nums = torch.cuda.device_count() # GPU数
    device_id = rank % args.gpu_nums

    if rank == 0: os.makedirs(args.result_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if rank == 0: logger = get_logger(args)

    # create model LoRAに対応
    model = MyModel(args).to(device_id)

    # # 1. 新しいトークンの追加
    # num_new_tokens = 10
    # new_embeddings = torch.FloatTensor(num_new_tokens, model.language_model.shared.embedding_dim).uniform_(-0.1, 0.1).to(device_id)
    # model.language_model.shared.weight.data = torch.cat([model.language_model.shared.weight.data, new_embeddings], 0)

    # # 2. 既存のトークンの重みの凍結
    # for param in model.language_model.shared.parameters():
    #     param.requires_grad = False

    # config = LoraConfig(
    #     r=16,
    #     lora_alpha=16,
    #     target_modules = ["language_model.shared", "language_model.encoder.block.*.layer.*.SelfAttention.q", "language_model.encoder.block.*.layer.*.SelfAttention.k", "language_model.encoder.block.*.layer.*.SelfAttention.v"],
    #     lora_dropout=0.1,
    #     bias="none",
    # )
    
    # print("凍結時")
    # print_trainable_parameters(model)
    # lora_model = get_peft_model(model, config)
    # print("LoRA")
    # print_trainable_parameters(lora_model)
    model = DDP(model, device_ids=[device_id])
    
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(args, optimizer)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True)
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True, extra_ids=0, additional_special_tokens =[f"<extra_id_{i}>" for i in range(100)] + [f"<loc_{i}>" for i in range(args.loc_vocab_size)] + [f"<img_{i}>" for i in range(args.image_vocab_size)])

    # データの設定
    train_dataset, val_dataset = get_data(args, src_tokenizer, tgt_tokenizer)
    train_loader = get_distributed_dataloader(args, train_dataset, shuffle=True)
    val_loader = get_distributed_dataloader(args, val_dataset, shuffle=False)

    if args.num_epochs is None:
        args.num_epochs = int(args.num_steps / len(train_loader)) + 1
    steps = 0
    min_val_loss = 100
    loss_counter = LossCounter()
    acc_counter = AccCounter()
    
    
    for epoch in range(1, args.num_epochs+1):
        # 学習ループ
        image_mask_ratio = 0.0
        if args.image_model_train:
            model.module.image_model.train()
        model.module.transformer.train()
        train_loss = torch.tensor(0.0).to(device_id)
        train_count = torch.tensor(0).to(device_id)
        train_correct_count = torch.tensor(0).to(device_id)  # 正解のカウント用の変数を初期化
        pbar = tqdm(total=int(np.ceil(len(train_loader)/args.accumulation_steps)), desc=f'Train (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))
        for i, (src_images, tgt_images, src_texts, tgt_texts) in enumerate(train_loader):
            tgt_text = tgt_texts
            if i % args.accumulation_steps == 0:
                optimizer.zero_grad()
            src_images = src_images.to(device_id, non_blocking=True)
            # if args.pretrain:
            #     tgt_images = tgt_images.to(device_id)
            #     tgt_texts, _ = model.module.image_to_z(tgt_images)
            src_texts = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(device_id, non_blocking=True) # ['pt', 'tf', 'np', 'jax']
            tgt_texts = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(device_id, non_blocking=True) # ['pt', 'tf', 'np', 'jax']
            # if rank == 0:
            #     # logger.info('yomikomi')
            loss,outputs = model(src_images, src_texts, tgt_texts, image_mask_ratio=image_mask_ratio)
            # if rank == 0:
            #     logger.info('ok')
            loss /= args.accumulation_steps
            loss.backward()

            train_loss += loss.item() * src_images.shape[0]
            train_count += src_images.shape[0]

            # args.accumulation_steps回の勾配を蓄積してから、optimizer.step()を呼び出す
            if (i + 1) % args.accumulation_steps == 0 or i + 1 == len(train_loader):
                optimizer.step()
                pbar.update(1)
                if rank == 0: steps += 1
                if args.num_epochs is None:
                    scheduler.step()
                    
            # 予測の取得
            preds = tgt_tokenizer.batch_decode(outputs[:, 1:-1])
            cleaned_preds = [pred.replace('</s>', '').replace('<pad>', '') for pred in preds]

            # 予測と実際のテキストが一致するかどうかを確認
            corrects = [1 if pred == actual else 0 for pred, actual in zip(cleaned_preds, tgt_text)]
            train_correct_count += sum(corrects)

            # print("pre",preds)
            # print("tgt",tgt_text)
            # print(correct_count)


        # 他のノードから集める
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_count, op=dist.ReduceOp.SUM)

        if rank == 0:
            train_loss /= train_count
            loss_counter.add("train", train_loss.cpu().numpy().copy())
            train_acc = train_correct_count.float() / train_count  # 正答率を計算
            acc_counter.add("train", train_acc.cpu().numpy().copy())  # 正答率をAccCounterに追加

        if args.lr_scheduler != '' and args.num_steps is None:
            scheduler.step()
        pbar.close()
        # 検証ループ
        if args.image_model_train:
            model.module.image_model.eval()
        model.module.transformer.eval()
        val_loss = torch.tensor(0.0).to(device_id)
        val_count = torch.tensor(0).to(device_id)
        val_correct_count = torch.tensor(0).to(device_id)  # 正解のカウント用の変数を初期化
        val_loop = tqdm(val_loader, desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))
        for src_images, tgt_images, src_texts, tgt_texts in val_loop:
            with torch.no_grad():
                tgt_text = tgt_texts
                src_images = src_images.to(device_id, non_blocking=True)
                # if args.pretrain:
                #    tgt_images = tgt_images.to(device_id)
                #    tgt_texts, _ = model.module.image_to_z(tgt_images)
                src_texts = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(device_id, non_blocking=True) # ['pt', 'tf', 'np', 'jax']
                tgt_texts = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(device_id, non_blocking=True) # ['pt', 'tf', 'np', 'jax']
                
                loss,outputs = model(src_images, src_texts, tgt_texts)
                
                val_loss += loss.item() * src_images.shape[0]
                val_count += src_images.shape[0]
                
                # 予測の取得
                preds = tgt_tokenizer.batch_decode(outputs[:, 1:-1])
                cleaned_preds = [pred.replace('</s>', '').replace('<pad>', '') for pred in preds]

                # 予測と実際のテキストが一致するかどうかを確認
                corrects = [1 if pred == actual else 0 for pred, actual in zip(cleaned_preds, tgt_text)]
                val_correct_count += sum(corrects)

        # 他のノードから集める
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_correct_count, op=dist.ReduceOp.SUM)

        if rank == 0:
            val_loss /= val_count
            loss_counter.add("val", val_loss.cpu().numpy().copy())
            val_acc = val_correct_count.float() / val_count  # 正答率を計算
            acc_counter.add("val", val_acc.cpu().numpy().copy())  # 正答率をAccCounterに追加

            logger.info(f'[Epoch ({epoch}/{args.num_epochs})] Train loss : {train_loss}, Val loss : {val_loss}, Train acc : {train_acc}, Val acc : {val_acc}, Steps : {steps}, train_count : {train_count},train_correct_count : {train_correct_count},val_count : {val_count},val_correct_count : {val_correct_count}')
        
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print('Best Model saving...')
                model.module.save()
                logger.info('Best Model saved')

            if args.save_interval is not None:
                if args.num_steps is None:
                    if (epoch) % args.save_interval == 0:
                        print(f'Model {epoch} saving...')
                        model.module.save(result_name=f'epoch_{epoch}.pth')
                        print(f'Model {epoch} saved')
                else:
                    if steps % args.save_interval == 0:
                        print(f'Model {steps} saving...')
                        model.module.save(result_name=f'step_{steps}.pth')
                        print(f'Model {steps} saved')
            
    if rank == 0: 
        loss_counter.plot_loss(args.result_dir)
        acc_counter.plot_loss(args.result_dir)  # result_dirは保存先のディレクトリを指定

if __name__=="__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    train()