#!/usr/bin/env bash

CNNDM_INPUT_PATH=../summarization/dataset/raw/CNN_Daily/cnndm
CNNDM_OUTPUT_PATH=../summarization/dataset/processed/CNN_Daily/cnndm

GIGA_INPUT_PATH=../summarization/dataset/raw/Gigaword/sumdata/train
GIGA_OUTPUT_PATH=../summarization/dataset/processed/Gigaword/sumdata/train

# processsing cnn daily data
function preprocess_cnndm() {
       nohup python preprocess.py -train_src "${CNNDM_INPUT_PATH}/train.txt.src" \
                            -train_tgt "${CNNDM_INPUT_PATH}/train.txt.tgt.tagged" \
                            -valid_src "${CNNDM_INPUT_PATH}/val.txt.src" \
                            -valid_tgt "${CNNDM_INPUT_PATH}/val.txt.tgt.tagged" \
                            -save_data "${CNNDM_OUTPUT_PATH}/CNNDM" \
                            -src_seq_length 10000 \
                            -tgt_seq_length 10000 \
                            -src_seq_length_trunc 400 \
                            -tgt_seq_length_trunc 100 \
                            -dynamic_dict \
                            -share_vocab \
                            -shard_size 100000 \
                            > logs/preprocess_cnndm.log &
}
       
 # processsing gigaword data
function preprocess_giga() {
       nohup python preprocess.py -train_src "${GIGA_INPUT_PATH}/train.article.txt" \
                            -train_tgt "${GIGA_INPUT_PATH}/train.title.txt" \
                            -valid_src "${GIGA_INPUT_PATH}/valid.article.filter.txt" \
                            -valid_tgt "${GIGA_INPUT_PATH}/valid.title.filter.txt" \
                            -save_data "${GIGA_OUTPUT_PATH}/GIGA" \
                            -src_seq_length 10000 \
                            -dynamic_dict \
                            -share_vocab \
                            -shard_size 100000 \
                            > logs/preprocess_giga.log &
}

data_name=$1
if [[ ${data_name} = "cnndm" ]];then
       preprocess_cnndm
elif [[ ${data_name} = "giga" ]];then
       preprocess_giga
else
       echo "Unknown Dataset name(${data_name})!!!"
fi