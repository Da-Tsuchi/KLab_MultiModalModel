batch_size=200
# dataset="cc3m cc12m imagenet inaturalist places365 redcaps sun397"

# データセットimagenet_21k
dataset="openimage_loc"
lr=2e-4

# transformerのパラメタ(層数を変更:デコーダーの層数を1層に変更)
d_model=768
d_ff=3072
d_kv=64
num_heads=12
enc=2
dec=12

torchrun --nnodes=1 --nproc_per_node=4 train.py \
        -l google/flan-t5-large \
        -i microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft \
        --ffn \
        --transformer_model_name="google/flan-t5-base" \
        --transformer_d_model $d_model \
        --transformer_d_ff $d_ff \
        --transformer_d_kv $d_kv \
        --transformer_num_heads $num_heads \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --lr $lr \
        --lr_scheduler 'LambdaLR' \
        --optimizer AdamW \
        -b $batch_size \
        --num_epochs 50 \
        --root_dir /data/ \
        --datasets $dataset \
        --result_dir results/0114/$lr/$dataset\/enc$enc\_dec$dec/ \
        --loss "CrossEntropy"\
        --loc_learn "lora"\
        --lora_r 16 \
        --lora_alpha 48 \
        # --lora_target_modules $target_modules \