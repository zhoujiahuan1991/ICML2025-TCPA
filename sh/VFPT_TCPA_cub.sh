#!/bin/bash

batch_size=128
mixup="mixup"
random_e=10
w=0.01
len_prompts=1



dataset="cub"
topk_cls=10
pool_size_cls=18
as=("10")
bs=("18")
cs=("0.0015")
ds=("0.5")

for topk in "${as[@]}"
do
    for pool_size in "${bs[@]}"
    do
        for lr in "${cs[@]}"
        do
            for wd in "${ds[@]}"
            do
                python main.py --info="${dataset}-lr=${lr}-wd=${wd}" \
                    --model=VFPT_TCPA --output_path="./Output_VFPT_TCPA" \
                    --pretrained=imagenet22k --dataset=$dataset \
                    --batch_size=$batch_size --lr=$lr --weight_decay=$wd --mixup=$mixup \
                    --len_prompts_cls=${len_prompts} --len_prompts_image=${len_prompts} \
                    --base_dir='/your/data/path' \
                    --topk_cls=$topk_cls --topk_image=$topk --pool_size_cls=${pool_size_cls} --pool_size_image=${pool_size} \
                    --RDVP --TDVP --IPrompt\
                    --pool_loss_w=$w --random_epoch=$random_e
            done
        done
    done
done