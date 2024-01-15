batch_size=30

epoch=50

# base用
d_model=768
d_ff=3072

d_kv=64
num_heads=12
enc=2
dec=12

dataset="openimage_loc"
python test.py \
        -l google/flan-t5-small \
        -i microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft \
        --ffn \
        -tm google/flan-t5-base \
        --transformer_d_model $d_model \
        --transformer_d_ff $d_ff \
        --transformer_d_kv $d_kv \
        --transformer_num_heads $num_heads \
        --transformer_num_layers $enc \
        --phase train \
        --loss CrossEntropy \
        -b $batch_size \
        --num_epochs $epoch \
        --datasets $dataset \
        --root_dir /user/data/ \
        --result_dir "inference/$dataset/lora/enc$enc\_dec$dec/Linear$epoch/" \
        --loc_learn "lora"\
        --pth_path "/home/tsuchida/KLab_MultiModalModel/results/lora/base/vg/visual_genome_refexp/enc2_dec12/epoch_50"

