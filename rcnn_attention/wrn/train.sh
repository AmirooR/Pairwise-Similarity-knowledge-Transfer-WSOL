CONFIG="./configs/${1}.config"
OUTPUT="./logs/${1}"

mkdir -p ${OUTPUT}
cp ${CONFIG} ${OUTPUT}
export CUDA_VISIBLE_DEVICES="1"
export TF_ENABLE_WINOGRAD_NONFUSED=1
python train.py --logtostderr \
                                         --pipeline_config_path=${CONFIG} \
                                         --train_dir="${OUTPUT}/train/" \
                                         --num_clones=1
