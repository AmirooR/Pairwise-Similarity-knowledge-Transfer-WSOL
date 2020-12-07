OUTPUT="./logs/${1}"
CONFIG="./configs/${1}.config"
EVALDIR="./logs/${1}/eval"
mkdir -p ${OUTPUT}
mkdir -p ${EVALDIR}
cp ${CONFIG} "${OUTPUT}/eval.config"

CUDA_VISIBLE_DEVICES="0", python eval.py --logtostderr \
                                         --checkpoint_dir="./${OUTPUT}/train" \
                                         --pipeline_config_path=$CONFIG \
                                         --eval_dir="./${OUTPUT}/eval" 
