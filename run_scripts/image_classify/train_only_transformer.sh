batch_size=256
# dataset="cc3m cc12m imagenet inaturalist places365 redcaps sun397"
dataset="imagenet_21k"

# transformerのパラメタ(層数を変更)
d_model=768
d_ff=3072
d_kv=64
num_heads=12
enc=2
dec=1

for model in "google/flan-t5-small"
do
# --language_model_name google/flan-t5-base \
# --transformer_model_name $model \
torchrun --nnodes=1 --nproc_per_node=4 train.py \
        -l google/flan-t5-large \
        -i microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft \
        --ffn \
        --transformer_d_model $d_model \
        --transformer_d_ff $d_ff \
        --transformer_d_kv $d_kv \
        --transformer_num_heads $num_heads \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --lr 0.001 \
        --optimizer AdamW \
        --batch_size $batch_size \
        --num_epochs 20 \
        --save_interval 50 \
        --root_dir /data01/ \
        --datasets $dataset \
        --result_dir results/image_classify/only_transformer/$model/
done
