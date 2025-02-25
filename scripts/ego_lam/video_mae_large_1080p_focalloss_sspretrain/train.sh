export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='scripts/ego_lam/video_mae_large_1080p_focalloss_sspretrain/'
DATA_PATH='ego4d/look-at-me/data/'
MODEL_PATH='model/vit_l_hybrid_pt_800e_ssv2_ft.pth'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

        
# batch_size can be adjusted according to the graphics card
srun -p $PARTITION \
        --open-mode append \
        -o ${OUTPUT_DIR}/output \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        --quotatype=auto \
        ${SRUN_ARGS} \
        python -u run_class_finetuning.py \
        --model vit_large_patch16_224 \
        --data_set  ego4d \
        --nb_classes 2 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 32 \
        --num_workers 4 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 1 \
        --num_frames 7 \
        --sampling_rate 4 \
        --opt adamw \
        --lr 4e-6 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 50 \
        --dist_eval \
        --warmup_epochs 10 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --focalloss \
        --enable_deepspeed \
        ${PY_ARGS} &
echo please tail -f ${OUTPUT_DIR}/output
tail -f  ${OUTPUT_DIR}/output

