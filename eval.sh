OUTPUT="./logs/${1}"
CONFIG="./configs/${1}.config"

mkdir -p ${OUTPUT}
cp ${CONFIG} "${OUTPUT}/eval.config"

while true
do
  CUDA_VISIBLE_DEVICES="0", python eval.py --logtostderr \
                                         --checkpoint_dir="./${OUTPUT}/train" \
                                         --pipeline_config_path=$CONFIG \
                                         --eval_dir="./${OUTPUT}/eval"
  echo 'Sleeping for 5 seconds'
  sleep 5;
done
