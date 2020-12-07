## Pairwise Similarity Knowledge Transfer for Weakly Supervised Object Localization
This repository is the official implementation of ECCV 2020 paper: [Pairwise Similarity Knowledge Transfer for Weakly Supervised Object Localization paper](https://arxiv.org/abs/2003.08375). It also includes the original implementation of the ICCV 2019 paper: [Learning to Find Common Objects Across Few Image Collections](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shaban_Learning_to_Find_Common_Objects_Across_Few_Image_Collections_ICCV_2019_paper.pdf).

## Requirements

This project is tested using python2.7 and tensorflow-1.4. Other dependencies are:
1. [tensorflow models](https://github.com/tensorflow/models/commit/3bf85a4eddb9c56a28cc266ee4aa5604fb4d8334). Note that this is the exact commit we used. However, newer commits may also work as well.
2. [tensorpack](https://github.com/amiroor/tensorpack). Note that this is a clone of a specific commit of the original tensorpack. Our code only works with this commit.
3. [OpenGM](https://github.com/opengm/opengm.git)  You need to compile the python extension along with external TRWS and add the compiled shared library to your python environment.

## Setup

1. Clone this repository.
```bash
git clone git@github.com:AmirooR/Pairwise-Similarity-knowledge-Transfer-WSOL.git
```
2. Add current folder(`Pairwise-Similarity-knowledge-Transfer-WSOL`), tensorflow_model's `research` `research/slim`, and `tensorpack` directories to your python path.
3. Copy the proto files in `rcnn_attention/protos/` directory in tensorflow model's `research/object_detection/protos` and follow their instructions to setup object detection api and compile the proto files with protobuf compiler.
4. You should have extracted inception resnet features and dataset split `.pkl` files in correct paths to run imagenet experiments. As an example look at [this line in this config file](https://github.com/AmirooR/Pairwise-Similarity-knowledge-Transfer-WSOL/blob/master/rcnn_attention/wrn/configs/mil/imagenet/inception_resnet/agnostic_model/agnostic_box_multi_fea/pairwise_loop/templates/k2_icm_301.config#L223). Contact us if you want the features and the dataset split files or need instructions on that.

## Training and evaluation
1. Train agnostic pairwise model on source split using [this config](https://github.com/AmirooR/Pairwise-Similarity-knowledge-Transfer-WSOL/blob/master/rcnn_attention/wrn/configs/mil/imagenet/inception_resnet/agnostic_model/agnostic_box_multi_fea/k2n0.config).
```bash
cd rcnn_attention/wrn
bash train.sh mil/imagenet/inception_resnet/agnostic_model/agnostic_box_multi_fea/k2n0
```
2. Warmup initialization: change directory to `rcnn_attention/wrn` folder and run the [`aggregate.sh`](https://github.com/AmirooR/Pairwise-Similarity-knowledge-Transfer-WSOL/blob/master/rcnn_attention/wrn/aggregate.sh) script. This will save `multifea_K8_init.pkl` dataset using Greedy Tree method by finding common object across groups of 8 images. Check the config files pointed in the script and set the correct paths in them.
```bash
cd rcnn_attention/wrn
bash aggregate.sh
```
3. Run multifold training, warmup with Greedy Inference, and ICM inference loop using [imagenet_multifold_train_and_evaluate_loop_icm.sh](https://github.com/AmirooR/Pairwise-Similarity-knowledge-Transfer-WSOL/blob/master/rcnn_attention/wrn/imagenet_multifold_train_and_evaluate_loop_icm.sh) script in `wrn` folder.
```bash
bash imagenet_multifold_train_and_evaluate_loop_icm.sh
```
This will save the resulting datasets and write the infos/evaluation in the respective log folder for each fold and iterations. 

## Cite
If you use this code, please cite our papers:

```
@inproceedings{rahimi20pairwise,
 author = {Rahimi, Amir and Shaban, Amirreza and Ajanthan, Thalaiyasingam and Hartley, Richard and Boots, Byron},
 booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
 title = {Pairwise Similarity Knowledge Transfer for Weakly Supervised Object Localization},
 year = {2020}
}
@inproceedings{shaban19learning,
 author = {Shaban, Amirreza* and Rahimi, Amir* and Bansal, Shray and Gould, Stephen and Boots, Byron and Hartley, Richard},
  booktitle = {Proceedings of the International Conference on Computer Vision ({ICCV})},
  title = {Learning to Find Common Objects Across Few Image Collections},
  year = {2019}
}
```
