#! /usr/bin/env bash
eval_device=0
cur_dir=$PWD
experiment_name="multifea_k8"
eval_config_template="configs/mil/imagenet/inception_resnet/agnostic_model/agnostic_box_multi_fea/k8n0.config"
train_dir="logs/mil/imagenet/inception_resnet/agnostic_model/agnostic_box_multi_fea/k2n0"
root="mil/imagenet/inception_resnet/agnostic_model/agnostic_box_multi_fea/step0"
configs_root="configs/${root}"
logs_root="logs/${root}"
aggregate_ncobj_proposals=500
num_aggregate_examples=25000 #65478
update_split="revisit_val1"

evaluate(){
  train_dir=$1
  eval_config_file=$2
  evaldir=$3
  CUDA_VISIBLE_DEVICES="${eval_device}" python eval.py --logtostderr \
                  --checkpoint_dir="${train_dir}/calib_train" \
                  --pipeline_config_path="${eval_config_file}" \
                  --eval_dir="${evaldir}" \
                  --calibration=true >> "${train_dir}/eval.log" 2>&1 
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
evaluate $train_dir $eval_config_file "${train_dir}/aggregate"
