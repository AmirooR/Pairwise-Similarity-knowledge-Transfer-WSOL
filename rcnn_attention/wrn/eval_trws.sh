#INPUT="det/greedy/k16_pw2"
#INPUT="wrn_det/k4/k4_n2_mrf"
#INPUT="iccv/mini/main/mrf/bs10/k4n0"
INPUT=$1
OUTPUT="./logs/${INPUT}"
CONFIG="./configs/${INPUT}.config"
EVALDIR="./logs/${INPUT}/eval"
mkdir -p ${OUTPUT}
mkdir -p ${EVALDIR}
cp ${CONFIG} "${OUTPUT}/eval.config"

CUDA_VISIBLE_DEVICES="1", python eval.py --logtostderr \
                                         --checkpoint_dir="./${OUTPUT}/train" \
                                         --pipeline_config_path=$CONFIG \
                                         --eval_dir="./${OUTPUT}/eval" \
                                         --add_mrf=true \
                                         --add_unaries=false \
                                         --add_brute_force=false \
                                         --unary_scale=2.3 \
                                         --mrf_type="dense_trws" #2>test.txt
                                         #> mini_k4_n0_dense_trws.txt

