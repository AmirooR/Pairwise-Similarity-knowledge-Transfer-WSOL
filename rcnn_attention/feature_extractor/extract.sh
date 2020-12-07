OUTPUT="./logs/${1}"
CONFIG="../configs/${1}.config"

mkdir -p ${OUTPUT}
cp ${CONFIG} "${OUTPUT}/eval.config"

CUDA_VISIBLE_DEVICES="1", python extract.py --logtostderr \
                                       --checkpoint_dir="../logs/${1}/train" \
                                       --pipeline_config_path=$CONFIG \
                                       --eval_dir="./${OUTPUT}/extract300_338k"
