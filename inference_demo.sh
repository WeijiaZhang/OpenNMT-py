#!/usr/bin/env bash

CNNDM_MODEL_PREFIX=../opennmt_saved_models/cnndm
CNNDM_TRANS_MODEL_PREFIX=../opennmt_saved_models/cnndm_transformer

CNNDM_TEST_IN_PATH=../summarization/dataset/raw/CNN_Daily/cnndm
CNNDM_TEST_OUT_PATH=../summarization/testout

function eval_cnndm() {
	model_step="step_$1.pt"
	gpu_idx=$2
	model_path="${CNNDM_MODEL_PREFIX}_${model_step}"
	CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python translate.py -gpu 0 \
                    -batch_size 16 \
                    -beam_size 5 \
                    -model ${model_path} \
                    -src "${CNNDM_TEST_IN_PATH}/test.txt.src" \
                    -output "${CNNDM_TEST_OUT_PATH}/cnndm_step_$1.out" \
                    -min_length 35 \
                    -verbose \
                    -stepwise_penalty \
                    -coverage_penalty summary \
                    -beta 5 \
                    -length_penalty wu \
                    -alpha 0.9 \
                    -verbose \
                    -block_ngram_repeat 3 \
                    -ignore_when_blocking "." "</t>" "<t>" \
                    > logs/eval_cnndm_demo.log &
}

function eval_cnndm_trans() {
    model_step="step_$1_backup.pt"
    gpu_idx=$2
    model_path="${CNNDM_TRANS_MODEL_PREFIX}_${model_step}"
    CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python translate.py -gpu 0 \
                    -batch_size 16 \
                    -beam_size 5 \
                    -model ${model_path} \
                    -src "${CNNDM_TEST_IN_PATH}/test.txt.src" \
                    -output "${CNNDM_TEST_OUT_PATH}/cnndm_trans_step_$1.out" \
                    -min_length 35 \
                    -verbose \
                    -stepwise_penalty \
                    -coverage_penalty summary \
                    -beta 5 \
                    -length_penalty wu \
                    -alpha 0.9 \
                    -verbose \
                    -block_ngram_repeat 3 \
                    -ignore_when_blocking "." "</t>" "<t>" \
                    > logs/eval_cnndm_trans_demo.log &
}


train_order=$1
step=$2
gpu_idx_global=$3
if [[ ${train_order} = "cnndm" ]]; then
        eval_cnndm  ${step}  ${gpu_idx_global}
elif [[ ${train_order} = "cnndm_trans" ]]; then
        eval_cnndm_trans  ${step}  ${gpu_idx_global}
else
        echo "Unknown Train Order (${train_order}) !!!"
fi