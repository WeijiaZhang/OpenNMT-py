#!/usr/bin/env bash

CNNDM_DATA_PREFIX=../summarization/dataset/processed/CNN_Daily/cnndm/CNNDM
CNNDM_MODEL_PREFIX=../opennmt_saved_models/cnndm
CNNDM_TRANS_MODEL_PREFIX=../opennmt_saved_models/cnndm_transformer

GIGA_DATA_PREFIX=../summarization/dataset/processed/Gigaword/sumdata/train/GIGA
GIGA_MODEL_PATH=../opennmt_saved_models/giga

function train_cnndm() {
        gpu_idx=$1
        CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python train.py -data ${CNNDM_DATA_PREFIX} \
                -save_model ${CNNDM_MODEL_PREFIX} \
                -copy_attn \
                -global_attention mlp \
                -word_vec_size 128 \
                -rnn_size 512 \
                -layers 1 \
                -encoder_type brnn \
                -max_grad_norm 2 \
                -dropout 0. \
                -batch_size 16 \
                -valid_batch_size 16 \
                -train_steps 200000 \
                -valid_steps 10000 \
                -save_checkpoint_steps 10000 \
                -keep_checkpoint 7 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -reuse_copy_attn \
                -copy_loss_by_seqlength \
                -bridge \
                -seed 777 \
                -world_size 1 \
                -gpu_ranks 0 \
                > logs/train_cnndm_demo.log &   
}

function train_cnndm_trans() {
        gpu_idx=$1
        CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python train.py -data ${CNNDM_DATA_PREFIX} \
                   -save_model ${CNNDM_TRANS_MODEL_PREFIX} \
                   -layers 4 \
                   -rnn_size 512 \
                   -word_vec_size 512 \
                   -optim adam \
                   -encoder_type transformer \
                   -decoder_type transformer \
                   -position_encoding \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -learning_rate 2 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 16 \
                   -valid_batch_size 16 \
                   -train_steps 200000 \
                   -valid_steps 10000 \
                   -save_checkpoint_steps 10000 \
                   -keep_checkpoint 7 \
                   -accum_count 4 \
                   -share_embeddings \
                   -copy_attn \
                   -param_init_glorot \
                   -world_size 1 \
                   -gpu_ranks 0 \
                   > logs/train_cnndm_trans_demo.log &   
}

function train_giga() {
        gpu_idx=$1
        CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python train.py -data ${GIGA_DATA_PREFIX} \
                -save_model ${GIGA_MODEL_PATH} \
                -copy_attn \
                -reuse_copy_attn \
                -train_steps 200000 \
                -save_checkpoint_steps 5000 \
                -keep_checkpoint 10 \
                -world_size 2 \
                -gpu_ranks 0 1 \
                > logs/train_giga_demo.log & 

}

train_order=$1
gpu_idx_global=$2
if [[ ${train_order} = "cnndm" ]]; then
        train_cnndm ${gpu_idx_global}
elif [[ ${train_order} = "cnndm_trans" ]]; then
        train_cnndm_trans ${gpu_idx_global}
elif [[ ${train_order} = "giga" ]]; then
        train_giga ${gpu_idx_global}
else
        echo "Unknown Train Order (${train_order}) !!!"
fi