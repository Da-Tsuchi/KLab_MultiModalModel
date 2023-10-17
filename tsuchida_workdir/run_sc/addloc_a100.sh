batch_size=64
# dataset="cc3m cc12m imagenet inaturalist places365 redcaps sun397"

# データセットimagenet_21k
dataset="openimage"

# transformerのパラメタ(層数を変更:デコーダーの層数を1層に変更)
d_model=768
d_ff=3072
d_kv=64
num_heads=12
enc=2
dec=12

target_modules="transformer.decoder.block.0.layer.0.SelfAttention.q"
# "transformer.decoder.block.*.layer.0.SelfAttention.k","transformer.decoder.block.*.layer.0.SelfAttention.v","transformer.decoder.block.*.layer.0.SelfAttention.o","transformer.decoder.block.*.layer.2.DenseReluDense.wi_0","transformer.decoder.block.*.layer.2.DenseReluDense.wi_1","transformer.decoder.block.*.layer.2.DenseReluDense.wo"]

# image:swin_large
# language:flant5 small

torchrun --nnodes=1 --nproc_per_node=4 train.py \
        -l google/flan-t5-small \
        -i microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft \
        --ffn \
        --transformer_model_name="google/flan-t5-base" \
        --transformer_d_model $d_model \
        --transformer_d_ff $d_ff \
        --transformer_d_kv $d_kv \
        --transformer_num_heads $num_heads \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --lr 1e-5 \
        --lr_scheduler 'LambdaLR' \
        --optimizer AdamW \
        -b $batch_size \
        --num_epochs 50 \
        --root_dir /user/data/ \
        --datasets $dataset \
        --result_dir results/loc/bf16/scratch/$dataset\/enc$enc\_dec$dec/ \
        --loss "CrossEntropy"\
        --loc_learn "train"\
        # --lora_target_modules $target_modules \