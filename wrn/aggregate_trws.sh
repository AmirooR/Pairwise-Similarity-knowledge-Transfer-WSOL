#! /usr/bin/env bash
eval_device=2
cur_dir=$PWD
experiment_name="energy_1"
#experiment_name="bcd_k2_trws_rand_test"
eval_config_template="configs/mil/coco/step0/energy_1.config"
#eval_config_template="configs/mil/coco/step0/k2n0_trws.config"
train_dir="logs/mil/coco/step0/energy_1"
root="mil/coco/energy/step0"
configs_root="configs/${root}"
logs_root="logs/${root}"
aggregate_ncobj_proposals=1
num_aggregate_examples=22 #8: 8239 #4: 23440 #16: 4760 #32:1212 #2: 46810 (10 epochs) #1: 46810 (10 k2 epochs)
update_split="energy_1"
ds_root="/mnt/scratch/amir/detection/amirreza/data/coco"


evaluate(){
  train_dir=$1
  eval_config_file=$2
  evaldir=$3
  experiment_name=$4
  CUDA_VISIBLE_DEVICES="${eval_device}" python eval.py --logtostderr \
                  --checkpoint_dir="${train_dir}/calib_train" \
                  --pipeline_config_path="${eval_config_file}" \
                  --eval_dir="${evaldir}" \
                  --calibration=true \
                  --add_mrf=true \
                  --add_unaries=false \
                  --unary_scale=-1.0 \
                  --mrf_type="dense_energy" >> "${evaldir}/${experiment_name}.log" 2>&1 

}


mkdir -p ${configs_root}
  
#create aggregate config (create save_split)
echo "CREATING AGGREGATE CONFIG"
save_split="${experiment_name}_init"
eval_config_file="${configs_root}/${experiment_name}_init_eval.config"
python config_utils.py --config_template_path ${eval_config_template} \
                       --is_training false --eval_num_examples ${num_aggregate_examples} \
                       --aggregate true --aggregate_save_split ${save_split} \
                       --aggregate_update_split ${update_split} \
                       --write_path ${eval_config_file} \
                       --ncobj_proposals ${aggregate_ncobj_proposals}

#aggregate
echo "PERFORMING AGGREGATION"
mkdir -p "${train_dir}/aggregate"
evaluate $train_dir $eval_config_file "${train_dir}/aggregate" $experiment_name
