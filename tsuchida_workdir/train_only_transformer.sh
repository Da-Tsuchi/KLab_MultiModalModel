batch_size=50
# dataset="cc3m cc12m imagenet inaturalist places365 redcaps sun397"
dataset="imagenet_21k"

# transformerのパラメタ
d_model=768
d_ff=3072
d_kv=64
num_heads=12
enc=2
dec=1

CUDA_VISIBLE_DEVICES=2 python /home/tsuchida/workspace/KLab_MultiModalModel/train_one.py \
        -l google/flan-t5-large \
        -i microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft \
        --ffn \
        --transformer_d_model $d_model \
        --transformer_d_ff $d_ff \
        --transformer_d_kv $d_kv \
        --transformer_num_heads $num_heads \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --pretrain \
        --lr 0.01 \
        --optimizer AdamW \
        --lr_scheduler StepLR \
        -b $batch_size \
        --num_epochs 20 \
        --root_dir /data01/ \
        --dataset $dataset \
        --result_dir results/pretrain/imagenet_21k/enc$enc\_dec$dec/
