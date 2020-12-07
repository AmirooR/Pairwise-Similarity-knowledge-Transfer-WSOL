OUTPUT="./logs/${1}"
CONFIG="./configs/${1}.config"
EVALDIR="./logs/${1}/calib_eval"
mkdir -p ${OUTPUT}
mkdir -p ${EVALDIR}
cp ${CONFIG} "${OUTPUT}/eval.config"

while true
do
  CUDA_VISIBLE_DEVICES="0", python eval.py --logtostderr \
                                         --checkpoint_dir="./${OUTPUT}/calib_train" \
                                         --pipeline_config_path=$CONFIG \
                                         --eval_dir="./${OUTPUT}/calib_eval" \
                                         --calibration=true
  echo 'Sleeping for 1 seconds';
  sleep 1;
done
