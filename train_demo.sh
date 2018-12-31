#!/usr/bin/env bash

CNNDM_DATA_PREFIX=../summarization/dataset/processed/CNN_Daily/cnndm
CNNDM_MODEL_PREFIX=../opennmt_saved_models

GIGA_DATA_PREFIX=../summarization/dataset/processed/Gigaword/sumdata/train/GIGA
GIGA_MODEL_PATH=../opennmt_saved_models/giga


function train_cnndm_no_tag() {
        gpu_idx=$1
        data_prefix="${CNNDM_DATA_PREFIX}/no_tag/CNNDM"
        model_prefix="${CNNDM_MODEL_PREFIX}/no_tag/cnndm"
        echo "train_cnndm_no_tag"
        CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python train.py -data ${data_prefix} \
                -save_model ${model_prefix} \
                -report_every 100 \
                -copy_attn \
                -reuse_copy_attn \
                -copy_loss_by_seqlength \
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
                -keep_checkpoint 20 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -bridge \
                -seed 777 \
                -world_size 1 \
                -gpu_ranks 0 \
                > logs/train_cnndm_no_tag_demo.log &   
}


function train_cnndm_no_tag_coverage() {
        gpu_idx=$1
        data_prefix="${CNNDM_DATA_PREFIX}/no_tag/CNNDM"
        model_prefix="${CNNDM_MODEL_PREFIX}/no_tag/cnndm"
        train_from_path=../opennmt_saved_models/no_tag/cnndm_step_130000_rouge2_27.30.pt
        echo "train_cnndm_no_tag_coverage"
        CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python train.py -data ${data_prefix} \
                -save_model ${model_prefix} \
                -report_every 100 \
                -train_from ${train_from_path} \
                -copy_attn \
                -reuse_copy_attn \
                -copy_loss_by_seqlength \
                -coverage_attn \
                -lambda_coverage 1 \
                -global_attention mlp \
                -bridge \
                -word_vec_size 128 \
                -rnn_size 512 \
                -layers 1 \
                -encoder_type brnn \
                -max_grad_norm 2 \
                -dropout 0. \
                -batch_size 16 \
                -valid_batch_size 16 \
                -train_steps 150000 \
                -valid_steps 2000 \
                -save_checkpoint_steps 2000 \
                -keep_checkpoint 10 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -seed 777 \
                -world_size 1 \
                -gpu_ranks 0 \
                > logs/train_cnndm_no_tag_coverage_demo.log &   
}

function train_cnndm_with_tag() {
        gpu_idx=$1
        data_prefix="${CNNDM_DATA_PREFIX}/with_tag/CNNDM"
        model_prefix="${CNNDM_MODEL_PREFIX}/with_tag/cnndm"
        echo "train_cnndm_with_tag"
        CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python train.py -data ${data_prefix} \
                -save_model ${model_prefix} \
                -report_every 100 \
                -with_tag \
                -copy_attn \
                -reuse_copy_attn \
                -copy_loss_by_seqlength \
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
                -keep_checkpoint 20 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -bridge \
                -seed 777 \
                -world_size 1 \
                -gpu_ranks 0 \
                > logs/train_cnndm_with_tag_demo.log &   
}


function train_cnndm_with_tag_coverage() {
        gpu_idx=$1
        data_prefix="${CNNDM_DATA_PREFIX}/with_tag/CNNDM"
        model_prefix="${CNNDM_MODEL_PREFIX}/with_tag/cnndm"
        train_from_path="${CNNDM_MODEL_PREFIX}/with_tag/cnndm_step_130000_rouge2_27.30.pt"
        echo "train_cnndm_with_tag_coverage"
        CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python train.py -data ${data_prefix} \
                -save_model ${model_prefix} \
                -report_every 100 \
                -train_from ${train_from_path} \
                -with_tag \
                -copy_attn \
                -reuse_copy_attn \
                -copy_loss_by_seqlength \
                -coverage_attn \
                -lambda_coverage 1 \
                -global_attention mlp \
                -word_vec_size 128 \
                -rnn_size 512 \
                -layers 1 \
                -encoder_type brnn \
                -max_grad_norm 2 \
                -dropout 0. \
                -batch_size 16 \
                -valid_batch_size 16 \
                -train_steps 150000 \
                -valid_steps 2000 \
                -save_checkpoint_steps 2000 \
                -keep_checkpoint 10 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -bridge \
                -seed 777 \
                -world_size 1 \
                -gpu_ranks 0 \
                > logs/train_cnndm_with_tag_coverage_demo.log &   
}

function train_cnndm_trans_with_tag() {
        gpu_idx=$1
        data_prefix="${CNNDM_DATA_PREFIX}/with_tag/CNNDM"
        model_prefix="${CNNDM_MODEL_PREFIX}/with_tag/cnndm_transformer"
        train_from_path=../opennmt_saved_models/cnndm_transformer_step_40000.pt
        echo "train_from_path"
        CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python train.py -data ${data_prefix} \
                   -save_model ${model_prefix} \
                   -report_every 100 \
                   -train_from ${train_from_path} \
                   -with_tag \
                   -copy_attn \
                   -layers 4 \
                   -rnn_size 512 \
                   -word_vec_size 512 \
                   -optim adam \
                   -encoder_type transformer \
                   -decoder_type transformer \
                   -position_encoding \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -learning_rate 2 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 16 \
                   -valid_batch_size 16 \
                   -train_steps 200000 \
                   -valid_steps 10000 \
                   -save_checkpoint_steps 10000 \
                   -keep_checkpoint 10 \
                   -accum_count 4 \
                   -share_embeddings \
                   -param_init_glorot \
                   -world_size 1 \
                   -gpu_ranks 0 \
                   > logs/train_cnndm_trans_with_tag_demo.log &   
                   # -warmup_steps 8000 \
}

function train_giga() {
        gpu_idx=$1
        echo "train_giga"
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
    train_cnndm_no_tag ${gpu_idx_global}

elif [[ ${train_order} = "cnndm_cvrg" ]]; then
    train_cnndm_no_tag_coverage ${gpu_idx_global}

elif [[ ${train_order} = "cnndm_tag" ]]; then
    train_cnndm_with_tag ${gpu_idx_global}

elif [[ ${train_order} = "cnndm_tag_cvrg" ]]; then
    train_cnndm_with_tag_coverage ${gpu_idx_global}

elif [[ ${train_order} = "cnndm_trans_tag" ]]; then
        train_cnndm_trans_with_tag ${gpu_idx_global}

elif [[ ${train_order} = "giga" ]]; then
        train_giga ${gpu_idx_global}
else
        echo "Unknown Train Order (${train_order}) !!!"
fi