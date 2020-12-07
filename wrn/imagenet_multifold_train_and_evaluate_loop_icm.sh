#! /usr/bin/env bash
eval_device=0
train_device=0
cur_dir=$PWD
experiment_name="icm_10fold_k8_init"
train_split="multifea_k8_init"
gt_split="alexfea300_revisit_val1"
train_config_template="configs/mil/imagenet/inception_resnet/agnostic_model/agnostic_box_multi_fea/pairwise_loop/templates/k2_301.config"
eval_config_template="configs/mil/imagenet/inception_resnet/agnostic_model/agnostic_box_multi_fea/pairwise_loop/templates/k8_transferred_301.config"
eval_config_template_icm="configs/mil/imagenet/inception_resnet/agnostic_model/agnostic_box_multi_fea/pairwise_loop/templates/k2_icm_301.config"
root="mil/inception_resnet/agnostic_model/agnostic_box_multi_fea/icm_10folds_k8_init"
configs_root="configs/${root}"
logs_root="logs/${root}"
eval_ncobj_proposals=500
aggregate_ncobj_proposals=500
num_loop_iters=2
train_iters=80000
num_eval_examples=1000
num_aggregate_examples=2500 
# 65000/10 = 6500  (folds might not be even use 8000 instead)
# icm uses k2 so divide by 2 (8000/2 = 4000)
num_icm_examples=4000
# use smaller number if you want to compute energy
icm_save_eval_freq=8000
ds_root="../feature_extractor/logs/imagenet/revisit_agnostic_box_multi_feas"
start_iter=1 #1
start_fold=0 #0
num_folds=10

evaluate_trws(){
  train_dir=$1
  eval_config_file=$2
  evaldir=$3
  method=$4
  CUDA_VISIBLE_DEVICES="${eval_device}" python eval.py --logtostderr \
                  --checkpoint_dir="${train_dir}/calib_train" \
                  --pipeline_config_path="${eval_config_file}" \
                  --eval_dir="${evaldir}" \
                  --calibration=true \
                  --add_mrf=true \
                  --add_unaries=true \
                  --unary_scale=2.0 \
                  --mrf_type="${method}" >> "${train_dir}/eval_${method}.log" 2>&1 
}

evaluate_greedy(){
  train_dir=$1
  eval_config_file=$2
  evaldir=$3
  CUDA_VISIBLE_DEVICES="${eval_device}" python eval.py --logtostderr \
                  --checkpoint_dir="${train_dir}/calib_train" \
                  --pipeline_config_path="${eval_config_file}" \
                  --eval_dir="${evaldir}" \
                  --calibration=true >> "${train_dir}/eval.log" 2>&1 
}

train_and_eval () {
  train_config_file=$1
  train_dir=$2
  checkpoint=$3  #last checkpoint file
  eval_config_file=$4
  evaldir="${train_dir}/calib_eval"
  echo $checkpoint
  cp $train_config_file $train_dir
  cp $eval_config_file "${train_dir}/calib_eval.config"
  #run training in background
  CUDA_VISIBLE_DEVICES="${train_device}" TF_ENABLE_WINOGRAD_NONFUSED=1 python train.py \
                  --logtostderr \
                  --pipeline_config_path="${train_config_file}" \
                  --train_dir="${train_dir}/train/" \
                  --num_clones=1 > "${train_dir}/train.log" 2>&1 &
  while :
  do
    # NOTE: Uncomment (next 3 commented lines) 
    #       if you want to evaluate during training
    #evaluate_greedy $train_dir $eval_config_file $evaldir
    sleep 20
    if [ -f $checkpoint ]; then
      echo "FOUND LAST ITERATION"
      #evaluate_greedy $train_dir $eval_config_file $evaldir
      #rm dataflow-pipe*
      break
    fi
  done
}


mkdir -p ${configs_root}
#loop
for iter in $(seq $start_iter $num_loop_iters);
do
  ##some complex logic for enabling continue from broken run
  starting_fold=0
  if [ $iter = $start_iter ];
  then
    if [ $start_fold -ge 1 ];
    then
      starting_fold=$start_fold
      last_fold=$((start_fold-1))
      train_split="${experiment_name}_${iter}_${last_fold}_ICM"
    fi

    if [ $iter -ge 2 -a $start_fold = 0 ];
    then
      last_iter=$((iter-1))
      train_split="${experiment_name}_${last_iter}_$((num_folds-1))_ICM"
    fi
  fi
  ##

  if [ ! $iter = $start_iter -o ! $start_fold -ge 1 ];
  then
    echo "Iteration ${iter}, ASSIGNING FOLDS..."
    python assign_folds.py --ds_root ${ds_root} --train_split ${train_split} --gt_split ${gt_split} --num_folds ${num_folds}
  fi
  
  
  for fold in $(seq $starting_fold $((num_folds-1)));
  do
    echo "Fold ${fold}" 
    #create train config
    echo "CREATING TRAIN CONFIG"
    train_config_file="${configs_root}/${experiment_name}_${iter}_${fold}.config"
    echo $train_config_file
    python config_utils.py --config_template_path ${train_config_template} \
                           --is_training true --split ${train_split} \
                           --train_num_steps ${train_iters} \
                           --write_path ${train_config_file} \
                           --eval_fold ${fold} --num_folds ${num_folds}
    #create train and calib eval folders
    echo "CREATING TRAIN AND CALIB EVAL FOLDERS"
    train_dir="${logs_root}/${experiment_name}_${iter}_${fold}"
    mkdir -p "${train_dir}/train"
    cd ${train_dir} && ln -s train calib_train
    cd $cur_dir
    
    #create eval config
    echo "CREATING EVAL CONFIG"
    eval_config_file="${configs_root}/${experiment_name}_${iter}_${fold}_eval.config"
    echo $eval_config_file
    python config_utils.py --config_template_path ${eval_config_template} \
                           --is_training false --eval_num_examples ${num_eval_examples} \
                           --aggregate false --write_path ${eval_config_file} \
                           --ncobj_proposals ${eval_ncobj_proposals} \
                           --eval_fold ${fold} --num_folds ${num_folds}
    
    #create eval folder
    echo "CREATING EVAL FOLDER"
    cd ${logs_root} && ln -s "${experiment_name}_${iter}_${fold}" "${experiment_name}_${iter}_${fold}_eval" 
    cd $cur_dir
    
    #perform train and eval
    echo "PERFORMING TRAINING AND EVALUATION"
    last_checkpoint_file="${train_dir}/train/model.ckpt-${train_iters}.index"
    train_log_file="${train_dir}/train.log"
    #cp ${init_checkpoint_folder}/* ${train_dir}/train/
    train_and_eval $train_config_file ${train_dir} $last_checkpoint_file $eval_config_file
    
    #create aggregate config (create save_split)
    echo "CREATING AGGREGATE CONFIG"
    save_split="${experiment_name}_${iter}_${fold}"
    update_split=${train_split}
    python config_utils.py --config_template_path ${eval_config_template} \
                           --is_training false --eval_num_examples ${num_aggregate_examples} \
                           --aggregate true --aggregate_save_split ${save_split} \
                           --aggregate_update_split ${update_split} \
                           --write_path ${eval_config_file} \
                           --ncobj_proposals ${aggregate_ncobj_proposals} \
                           --eval_fold ${fold} --num_folds ${num_folds}

    #aggregate
    echo "PERFORMING INITIALIZATION AGGREGATION"
    evaluate_greedy $train_dir $eval_config_file "${train_dir}/aggregate"

    #use save split to create bcd labelling info
    
    bcd_info="${save_split}_labelling_info"
    echo "CREATING LABELLING INFO: ${bcd_info}"
    python name2fea.py --split ${save_split} --save_name ${bcd_info} --ds_root ${ds_root} --eval_fold ${fold} --doublefeas true

    bcd_init_file="${ds_root}/ImageSet/${bcd_info}.pkl"
    old_save_split=${save_split}
    save_split="${save_split}_ICM"

    #create icm config with bcd_init
    python config_utils.py --config_template_path ${eval_config_template_icm} \
                           --is_training false --eval_num_examples ${num_icm_examples} \
                           --aggregate true --aggregate_save_split ${save_split} \
                           --aggregate_update_split ${old_save_split} \
                           --write_path ${eval_config_file} \
                           --ncobj_proposals ${aggregate_ncobj_proposals} \
                           --eval_fold ${fold} --num_folds ${num_folds} \
                           --bcd_init ${bcd_init_file} --k_shot 1 \
                           --save_eval_freq ${icm_save_eval_freq}

    #perform ICM
    echo "Performing ICM"
    evaluate_trws $train_dir $eval_config_file "${train_dir}/icm" "bcd_dense_trws"

    train_split=$save_split
    echo "ITERATION ${iter} DONE"
  done
done
