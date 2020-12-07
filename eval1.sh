OUTPUT="./logs/${1}"
CONFIG="./configs/${1}.config"

mkdir -p ${OUTPUT}
cp ${CONFIG} "${OUTPUT}/eval.config"

#TF_CPP_MIN_LOG_LEVEL="2", 
CUDA_VISIBLE_DEVICES="1", python eval.py --logtostderr \
                                         --checkpoint_dir="./${OUTPUT}/train" \
                                         --pipeline_config_path=$CONFIG \
                                         --eval_dir="./${OUTPUT}/eval"
