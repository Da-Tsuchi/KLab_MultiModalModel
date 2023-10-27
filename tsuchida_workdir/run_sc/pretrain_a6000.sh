batch_size=150
# dataset="cc3m cc12m imagenet inaturalist places365 redcaps sun397"

# データセットimagenet_21k
dataset="cc3m"

# transformerのパラメタ(層数を変更:デコーダーの層数を1層に変更)
d_model=768
d_ff=2048
d_kv=64
num_heads=12
enc=2
dec=12

# image:swin_large
# language:flant5 small

torchrun --nnodes=1 --nproc_per_node=4 train_decode.py \
        -l google/flan-t5-small \
        -i microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft \
        --ffn \
        -tm google/flan-t5-base \
        --transformer_d_model $d_model \
        --transformer_d_ff $d_ff \
        --transformer_d_kv $d_kv \
        --transformer_num_heads $num_heads \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --lr 1e-5 \
        --lr_scheduler 'StepLR' \
        --optimizer AdamW \
        -b $batch_size \
        --num_epochs 100 \
        --root_dir /user/data/ \
        --datasets $dataset \
        --result_dir results/a6000/lr1e-5/$dataset\/enc$enc\_dec$dec/ \
        --loss "CrossEntropy" \
