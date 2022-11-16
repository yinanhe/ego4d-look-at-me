#!/bin/bash
while [ ! -f /mnt/lustre/zhaozhiyu.vendor/videoMAEOUT/K400_opflow_rgb_pretrain_flowwithnorm_variance_align/flow_k400_800e.pth ]
do
sleep 60s
done
bash ./scripts_kinetics_finetune/roll_fintune_224_lr7e_4_flow.sh
