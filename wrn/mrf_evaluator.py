"""CoLocDetection model evaluator.

This file provides a generic evaluation method that can be used to evaluate a
CoLocDetectionModel.
"""
import functools
import logging
import tensorflow as tf

from rcnn_attention import eval_util as eval_util
from rcnn_attention import evaluator as det_evaluator
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import prefetcher
from rcnn_attention.coloc import standard_fields
fields = standard_fields.CoLocInputDataFields
from object_detection.utils import ops
import numpy as np
from IPython import embed
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

import itertools
from tensorflow.python.platform import flags
from mrf.mrf_evaluation import extend_for_mrf, add_mrf_scores
from calibration.calibration_utils import add_calibration_functions
import time
import pickle

FLAGS = flags.FLAGS

slim = tf.contrib.slim

EVAL_METRICS_FN_DICT = det_evaluator.EVAL_METRICS_FN_DICT

def read_pickle(path):
  with open(path, 'rb') as f:
    return pickle.load(f)


def is_multipart(input_k_shot, bag_size):
  if FLAGS.add_mrf and input_k_shot*input_k_shot*bag_size > 25000:
    return True
  if FLAGS.add_mrf and input_k_shot > 4 and bag_size > 50:
    return True
  return False

def get_max_batch(total_problems, bag_size):
  if bag_size == 1:
    if total_problems > 100000:
      m = 100000
    elif total_problems > 10000:
      m = 10000
    elif total_problems > 1000:
      m = 1000
    elif total_problems > 100:
      m = 100
    else:
      m = 6
  else:
    m = 6
  #max batch is 6
  for i in range(m,0,-1):
    if total_problems % i == 0:
      return i

def add_groundtruth(tensor_dict, input_dict,
                                 input_k_shot,
                                 bag_size,
                                 num_negative_bags,
                                 groundtruth_classes_list,
                                 original_groundtruth_classes_list):
  tensor_dict['input_k_shot'] = tf.constant(input_k_shot)
  tensor_dict['num_negative_bags'] = tf.constant(num_negative_bags)
  tensor_dict['bag_size'] = tf.constant(bag_size)

  groundtruth_classes = tf.concat(groundtruth_classes_list[:input_k_shot*bag_size], 0)
  groundtruth_classes = tf.reshape(groundtruth_classes, [input_k_shot, bag_size])
  original_groundtruth_classes = tf.concat(
      original_groundtruth_classes_list[:input_k_shot*bag_size],0)
  original_groundtruth_classes = tf.reshape(original_groundtruth_classes,
                                                      [input_k_shot, bag_size])
  if not fields.original_images in input_dict:
    raise Exception('Please add this tensors to mini-imagenet model, guzu.' \
                    ' It is added but ridam passe kallat')

  tensor_dict['original_image'] = input_dict[fields.original_images]
  tensor_dict['filenames'] = input_dict['filename']
  ndict = dict(boxes=[], classes=[], negative_classes=[])
  #ndict['target_class'] = input_dict[fields.groundtruth_target_class]
  total_bags = input_k_shot + num_negative_bags
  for i in range(total_bags): #ignore negative part for groundtruth
    if i < input_k_shot:
      ndict['boxes'].append(
          input_dict[fields.groundtruth_image_boxes+'_{}'.format(i)])
      ndict['classes'].append(input_dict[
          fields.groundtruth_image_classes+'_{}'.format(i)])
    else:
      ndict['negative_classes'].append(input_dict[
          fields.groundtruth_image_classes+'_{}'.format(i)])

  #ndict['fea_groundtruth_classes'] = groundtruth_classes
  #ndict['fea_original_groundtruth_classes'] = original_groundtruth_classes
  #ndict['fea_boxes'] = input_dict[fields.groundtruth_boxes]
  tensor_dict['groundtruth'] = ndict

def get_split(tensor, tf_ranges, sel, l):
  cur_tensor = tf.gather(tensor, tf.range(tf_ranges[sel][0],tf_ranges[sel][1]))
  shape = cur_tensor.shape.as_list()
  shape[0] = l
  cur_tensor.set_shape(shape)
  return cur_tensor

def wrap_with_variable(tensor, i):
  var = tf.get_variable(tensor.op.name + '_ref{}'.format(i),
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                        trainable=False)

  op = var.assign(tensor)

  return var, op

def get_tensor_dict(model, images_list,
                           boxes_list,
                           one_hot_gt_classes_list,
                           proposal_objectness_list,
                           tf_ranges,
                           l,
                           bag_size):
  images = tf.concat(images_list, 0)
  boxes = tf.concat(boxes_list, 0)
  one_hot_gt_classes = tf.concat(one_hot_gt_classes_list, 0)

  #images, op0 = wrap_with_variable(images)
  #boxes, op1 = wrap_with_variable(boxes)
  #one_hot_gt_classes, op2 = wrap_with_variable(one_hot_gt_classes)
  #reload_op = tf.group(op0, op1, op2)
  sel = tf.placeholder(tf.int32, ())
  cur_images = get_split(images, tf_ranges, sel, l)
  cur_boxes = get_split(boxes, tf_ranges, sel, l)
  cur_one_hot_gt_classes = get_split(one_hot_gt_classes,
                                     tf_ranges, sel, l)

  if proposal_objectness_list is not None:
    proposal_objectness = tf.concat(proposal_objectness_list,0)
    cur_proposal_objectness = get_split(proposal_objectness,
                                        tf_ranges, sel, l)

  float_images = tf.to_float(cur_images)

  model.provide_groundtruth(tf.split(cur_boxes, l),
                            tf.split(cur_one_hot_gt_classes, l),
                            None)

  if proposal_objectness_list is not None:
    model._groundtruth_lists['objectness_score'] = tf.reshape(cur_proposal_objectness, [-1, bag_size])


  prediction_dict = model.predict(float_images)
  detections = model.postprocess(prediction_dict)

  tensor_dict = dict()
  tensor_dict.update(model._tree_debug_tensors(
      include_k2_cross_similarty=FLAGS.add_mrf,
      apply_sigmoid_to_scores=False))

  label_id_offset = 1
  # Convert to the absolute coordinates
  for key,val in tensor_dict.items():
    if isinstance(val, dict):
      for mkey in val:
        if 'classes' in mkey:
          val[mkey] = val[mkey] + label_id_offset

  tensor_dict['sel'] = sel
  #tensor_dict['reload_op'] = reload_op
  return tensor_dict


def wrap_dict(d):
  with tf.variable_scope("", reuse=False):
    new_dict = {}
    ops = []
    for i,(k,v) in enumerate(d.iteritems()):
      var, op = wrap_with_variable(v, i)
      new_dict[k] = var
      ops.append(op)
    return new_dict, ops

def _extract_prediction_tensors(model,
                                input_k_shot,
                                create_input_dict_fn,
                                ignore_groundtruth=False,
                                use_target_class_in_predictions=False,
                                bcd_outside_images=0):
  """Restores the model in a tensorflow session.

  Args:
    model: model to perform predictions with.
    create_input_dict_fn: function to create input tensor dictionaries.
    ignore_groundtruth: whether groundtruth should be ignored.
    bcd_outside_images: maximum number of images outside the subproblem.
                        We expect to get one feature from each bag outside
                        the subproblem.
  Returns:
    tensor_dict: A tensor dictionary with evaluations.
  """
  tree_k_shot = model._k_shot
  bag_size = model._bag_size
  num_classes = model._num_classes
  num_negative_bags = model._num_negative_bags
  total_images = num_negative_bags + input_k_shot
  input_dict, thread = create_input_dict_fn()
  prefetch_queue = prefetcher.prefetch(input_dict, capacity=10)
  input_dict = prefetch_queue.dequeue()
  extra_dict = {}
  reduced_dict = {}

  ## Append additional variables for the outside bag features
  num_bcd_bags = int(np.ceil(bcd_outside_images/float(bag_size)))
  bcd_bags = None
  bcd_reload_op = None
  field_bcd_supportset = 'bcd_supportset'
  if bcd_outside_images > 0:
    assert(num_negative_bags == 0)
    assert('dense' in FLAGS.mrf_type)
    shape = [num_bcd_bags] + input_dict[fields.supportset].shape.as_list()[1:]
    #bcd_supportset = tf.placeholder(tf.float32, shape=shape,
                                       #name=field_bcd_supportset)
    bcd_supportset = tf.constant(0.0, dtype=tf.float32, shape=shape,
                                 name=field_bcd_supportset)
    bcd_supportset_var, bcd_reload_op = wrap_with_variable(bcd_supportset, 0)
    input_dict[field_bcd_supportset] = bcd_supportset_var
  ##

  for k,v in input_dict.iteritems():
    #if 'groundtruth_image' in k or 'original_images' in
    #k or 'target_class' in k or 'filename' in k:
    if ('groundtruth_image' in k or 'original_images' in k  or
        'filename' in k or field_bcd_supportset in k):
      extra_dict[k] = v
    else:
      #######
      #if k == fields.supportset:
        #v = tf.zeros_like(v)
        #embed()
      #####
      reduced_dict[k] = v
  #reduced_dict.pop('filename')
  input_dict, ops = wrap_dict(reduced_dict)
  input_dict.update(extra_dict)
  reload_op = tf.group(*ops)
  def _r(t):
    return tf.reshape(t, [-1,]+t.shape[2:].as_list())

  def reorder_for_k2(t):
    '''
    when tree_k_shot is 2, and we have negative bags and input_k_shot is greater than 2,
    this function reorders the list of inputs in a way that it puts the negative elements
    after every 2*bag_size elements.
    Notes:
      - assumes positive elements are in first bag_size*input_k_shot elements of the input list
      and the rest of the elements are negative bags i.e., num_negative_bags*bag_size last
      elements of the list.
    '''
    if num_negative_bags > 0 and tree_k_shot == 2 and input_k_shot != 2:
      pos_part = t[:bag_size*input_k_shot]
      neg_part = t[bag_size*input_k_shot:]
      assert input_k_shot % 2 == 0, 'input_k_shot should be multiple of 2'
      l = [pos_part[bag_size*2*i:bag_size*2*(i+1)] + neg_part for i in range(input_k_shot/2)]
      t = [i for x in l for i in x]
    return t

  def _s(t):
    return reorder_for_k2(tf.split(_r(t), total_images*bag_size))


  images_list = _s(input_dict[fields.supportset])
  groundtruth_classes_list = _s(input_dict[fields.groundtruth_classes])
  original_groundtruth_classes = input_dict[fields.original_groundtruth_classes]
  original_groundtruth_classes_list = _s(original_groundtruth_classes)
  one_hot_gt_classes_list = _s(tf.one_hot(tf.to_int32(original_groundtruth_classes),
                               num_classes + 1))
  boxes_list = _s(input_dict[fields.groundtruth_boxes])
  proposal_objectness_list = input_dict.get(fields.proposal_objectness, None)
  if proposal_objectness_list is not None:
    proposal_objectness_list = _s(proposal_objectness_list)
  if use_target_class_in_predictions:
    target_class = input_dict[fields.groundtruth_target_class]
    model._groundtruth_lists['target_class'] = target_class[tf.newaxis]

  bcd_bags = None
  if field_bcd_supportset in input_dict:
    bcd_bags = tf.split(_r(input_dict[field_bcd_supportset]), num_bcd_bags*bag_size)
  if FLAGS.add_mrf:
    lists = [images_list,
             groundtruth_classes_list,
             original_groundtruth_classes_list,
             one_hot_gt_classes_list,
             boxes_list]
    if proposal_objectness_list is not None:
      lists.append(proposal_objectness_list)
    extend_for_mrf(lists,
                   bcd_bags,
                   input_k_shot,
                   bag_size,
                   num_negative_bags)
    if is_multipart(input_k_shot+num_bcd_bags, bag_size):
      problem_size = (2+num_negative_bags)*bag_size ###TODO for K1 and BCD: add min(2, input_k_shot)
      total_problems = len(images_list)/problem_size
      max_batch = get_max_batch(total_problems, bag_size)
      assert total_problems % max_batch == 0, 'problems should be divisible by max_batch'

      batch_indices = range(total_problems//max_batch)
      l = max_batch*problem_size
      #start and end indices
      se = [(l*i,l*(i+1)) for i in batch_indices]
      #if total_problems % max_batch != 0:
      #  se += [(se[-1][1], len(images_list))]
      np_ranges = np.array(se)
      tf_ranges = tf.constant(np_ranges)
      tensor_dict = get_tensor_dict(model, images_list,
                                    boxes_list,
                                    one_hot_gt_classes_list,
                                    proposal_objectness_list,
                                    tf_ranges,
                                    l,
                                    bag_size)
      gt_dict = {}
      gt_dict['num_sel'] = len(se)
      add_groundtruth(gt_dict,
          input_dict,
          input_k_shot,
          bag_size,
          num_negative_bags,
          groundtruth_classes_list,
          original_groundtruth_classes_list)
      tensor_dict['reload_op'] = reload_op
      if bcd_reload_op is not None:
        tensor_dict['bcd_reload_op'] = bcd_reload_op
        tensor_dict['bcd_supportset_placeholder'] = bcd_supportset
      tensor_dict[fields.supportset] = input_dict[fields.supportset]
      tensor_dict['target_class'] = input_dict[fields.groundtruth_target_class]
      return [tensor_dict, gt_dict], thread

  tf_ranges = tf.constant([(0, len(images_list))])
  tensor_dict = get_tensor_dict(model, images_list,
                                       boxes_list,
                                       one_hot_gt_classes_list,
                                       proposal_objectness_list,
                                       tf_ranges,
                                       len(images_list),
                                       bag_size)

  gt_dict = {}
  gt_dict['num_sel'] = 1
  add_groundtruth(gt_dict, input_dict,
                                 input_k_shot,
                                 bag_size,
                                 num_negative_bags,
                                 groundtruth_classes_list,
                                 original_groundtruth_classes_list)
  tensor_dict['reload_op'] = reload_op
  if bcd_reload_op is not None:
    tensor_dict['bcd_reload_op'] = bcd_reload_op
    tensor_dict['bcd_supportset_placeholder'] = bcd_supportset
  tensor_dict[fields.supportset] = input_dict[fields.supportset]
  tensor_dict['target_class'] = input_dict[fields.groundtruth_target_class]
  return [tensor_dict,gt_dict], thread


def evaluate(create_input_dict_fn_list, input_config_names,
             create_model_fn, eval_config, categories,
             checkpoint_dir, eval_dir, input_k_shot):
  """Evaluation function for detection models.

  Args:
    create_input_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel.
    eval_config: a eval_pb2.EvalConfig protobuf.
    categories: a list of category dictionaries. Each dict in the list should
                have an integer 'id' field and string 'name' field.
    checkpoint_dir: directory to load the checkpoints to evaluate from.
    eval_dir: directory to write evaluation metrics summary to.
  """

  if eval_config.ignore_groundtruth and not eval_config.export_path:
    logging.fatal('If ignore_groundtruth=True then an export_path is '
                  'required. Aborting!!!')

  bcd_outside_images = 0
  label_feas_info = None
  if eval_config.HasField('bcd_init') and eval_config.bcd_init:
    label_feas_info = read_pickle(eval_config.bcd_init)
    bcd_outside_images=max([len(c) for c in label_feas_info['class2names'
                            ].values()])

  extract_prediction_tensors_fn_list = [
      lambda create_input_dict_fn=create_input_dict_fn:
      _extract_prediction_tensors(
        model=create_model_fn(),
        input_k_shot=input_k_shot,
        create_input_dict_fn=create_input_dict_fn,
        ignore_groundtruth=eval_config.ignore_groundtruth,
        use_target_class_in_predictions=eval_config.use_target_class_in_predictions,
        bcd_outside_images=bcd_outside_images)
      for create_input_dict_fn in create_input_dict_fn_list]

  required_batch_indices = False
  dummy_result_dict = None
  saved_results = []
  saved_energies = []

  def _process_batch(tensor_dicts, sess, batch_index, counters,
                     update_op, model_name=None):
    global dummy_result_dict
    """Evaluates tensors in tensor_dict, visualizing the first K examples.

    This function calls sess.run on tensor_dict, evaluating the original_image
    tensor only on the first K examples and visualizing detections overlaid
    on this original_image.

    Args:
      tensor_dict: a dictionary of tensors
      sess: tensorflow session
      b#atch_index: the index of the batch amongst all batches in the run.
      counters: a dictionary holding 'success' and 'skipped' fields which can
        be updated to keep track of number of successful and failed runs,
        respectively.  If these fields are not updated, then the success/skipped
        counter values shown at the end of evaluation will be incorrect.
      update_op: An update op that has to be run along with output tensors. For
        example this could be an op to compute statistics for slim metrics.

    Returns:
      result_dict: a dictionary of numpy arrays
    """
    import numpy as np
    localtime = time.localtime()
    #print("[{},{},{}]: Processing batch {}".format(localtime.tm_hour,
    #localtime.tm_min,  localtime.tm_sec, batch_index))
    def filter_dict(d, names):
      d = {k:v for (k,v) in d.iteritems()
                     if not k in names}
      return d

    [tensor_dict, gt_dict] = tensor_dicts
    num_sel = gt_dict['num_sel']
    gt_dict = filter_dict(gt_dict, ['num_sel'])
    sel = tensor_dict['sel']
    bcd_reload_op = tensor_dict.get('bcd_reload_op', None)
    bcd_supportset_placeholder = tensor_dict.get('bcd_supportset_placeholder', None)
    supportset = tensor_dict.get(fields.supportset, None)
    reload_op = tensor_dict['reload_op']
    tensor_dict = filter_dict(tensor_dict, ['sel', 'reload_op',
                                            'bcd_reload_op',
                                            'bcd_supportset_placeholder',
                                            fields.supportset])
    _, np_gt_dict = sess.run([reload_op, gt_dict])
    filenames = np_gt_dict.pop('filenames', None)
    bag_size = np_gt_dict['bag_size']
    input_k_shot = np_gt_dict['input_k_shot']
    ## set bcd bags here
    if 'bcd' in FLAGS.mrf_type:
      assert(filenames is not None)
      # read target_class
      target_class = int(sess.run(tensor_dict['target_class']))

      # compute bcd bag features
      class_image_names = label_feas_info['class2names'][target_class]
      outside_image_names = class_image_names - set(filenames[:input_k_shot])

      feas = []
      for name in outside_image_names:
        fea = label_feas_info['name2class2feas'][name][target_class]
        feas.append(fea)
      #[F,1,1,D]
      fea0 = np.stack(feas)
      num_bcd_feas = fea0.shape[0]
      to_pad_size = int(np.prod(bcd_supportset_placeholder.shape[:2]))
      pad_size = to_pad_size - num_bcd_feas
      assert(pad_size >= 0)
      if pad_size > 0:
        fea0 = np.pad(fea0, ((0, pad_size),(0,0),(0,0),(0,0)), 'constant')
        fea0 = np.reshape(fea0, (-1,bag_size,)+fea0.shape[1:])
      # set bcd bag variables
      sess.run(bcd_reload_op, {bcd_supportset_placeholder:fea0})
    ##

    if required_batch_indices:
      if batch_index > 0 and batch_index not in required_batch_indices:
        return dummy_result_dict
    if batch_index >= eval_config.num_visualizations:
      gt_dict.pop('original_image', None)
    def filter_nones(d):
      d = dict(filter(lambda item: item[1] is not None, d.items()))
      for k, v in d.iteritems():
        if type(v) == dict:
          d[k] = filter_nones(d[k])
      return d
    tensor_dict = filter_nones(tensor_dict)
    t0 = time.time()
    def join(r, r_part):
      for k,v in r_part.iteritems():
        if type(v) == np.ndarray:
          r[k] = np.vstack([r[k], v])
        elif type(v) == dict:
          join(r[k], v)
        elif type(v) == list:
          r[k] = r[k] + v
    #tensor_dict = filter_dict(tensor_dict)
    result_dict = np_gt_dict
    result_dicts = [sess.run(tensor_dict, {sel:i}) for i in range(num_sel)]
    result_dict.update(result_dicts[0])
    for i in range(1,num_sel):
      join(result_dict, result_dicts[i])
    forward_time = time.time() - t0
    meta_info = result_dict.pop('meta_info', dict())
    result_dict.pop('input_k_shot')
    meta_info['positive_images_name'] = filenames[:input_k_shot]
    num_negative_bags = result_dict.pop('num_negative_bags')
    bag_size = result_dict.pop('bag_size')
    localtime = time.localtime()
    #print("[{},{},{}]: FORWARD DONE".format(localtime.tm_hour,localtime.tm_min,  localtime.tm_sec))
    if FLAGS.add_mrf:
      #if batch_index >= 400 and (batch_index % 100) == 0:
      assert 'k2_cross_similarity_scores' in meta_info.keys(), 'key not in meta_info'
      functions = [lambda x: (x, 'identity')]
      functions.extend(add_calibration_functions(is_logit=True))
      #scales   = np.linspace(0.6,0.9,4).astype(np.float32)
      scales   = np.array([0.7], dtype=np.float32)
      if FLAGS.unary_scale >= 0.0:
        scales = np.array([FLAGS.unary_scale], dtype=np.float32)
      if 'bcd' in FLAGS.mrf_type:
        dense_part = (input_k_shot)*(input_k_shot-1)//2
        # [bcd_max_bags*k, bag_size*bag_size, 1]
        bcd_part = meta_info['k2_cross_similarity_scores'][dense_part:]
        # [bcd_max_bags*k, bag_size, bag_size]
        bcd_part = bcd_part.reshape((-1, bag_size,bag_size))
        if FLAGS.zero_pairwise:
          meta_info['k2_cross_similarity_scores'][:dense_part] = np.zeros_like(
              meta_info['k2_cross_similarity_scores'][:dense_part])

        ## FOR TEST
        #bcd_pairwises = np.zeros((to_pad_size, input_k_shot, bag_size), dtype=np.float32)
        #for i in range(bcd_part.shape[0]):
        #  k_num = i % input_k_shot
        #  bcd_bag_num = i / input_k_shot
        #  bcd_pairwises[bcd_bag_num*bag_size: (bcd_bag_num + 1)*bag_size, k_num, :] = bcd_part[i,:,:]
        #bcd_pairwises = bcd_pairwises[:num_bcd_feas]
        ## END

        bcd_part = bcd_part.reshape((-1, input_k_shot, bag_size, bag_size))
        bcd_pairwises1 = np.transpose(bcd_part, axes=(0,2,1,3))
        bcd_pairwises1 = bcd_pairwises1.reshape((-1, input_k_shot,
                                               bag_size))[:num_bcd_feas]
        meta_info['bcd_pairwises'] = bcd_pairwises1
        assert(len(scales) == 1)
      for f, p in itertools.product(functions, scales):
        res, inf_time, energy, mrf_name, model = add_mrf_scores(result_dict,
                                                                meta_info,
                                                                f, p, input_k_shot, filenames,
                                                                return_model=True)

        if 'energy' in FLAGS.mrf_type:
          logging.info("ENERGY,{},target_class,{}".format(energy, result_dict['target_class']))
          def get_method_epoch(name):
            name_fields = name.split('_')
            epoch = int(name_fields[-2])
            method = name_fields[3]
            return epoch, method
          epoch, method = get_method_epoch(filenames[0])
          saved_energies.append([method,epoch,energy])
          np.save(FLAGS.energy_save_name +'.npy', np.array(saved_energies))
          return None

        ## Update label_feas_info['name2class2feas'] with respect to the new selected
        ## labels
        #print(batch_index+1, supportset_np.std())
        if 'bcd' in FLAGS.mrf_type:
          supportset_np = sess.run(supportset)
          selected_labels = res['scores'].argmax(1)
          pinds = result_dict['Tree_K1']['proposal_inds'][:input_k_shot]

          #### DEBUG
          if False:
            gm = model._debug['gm']
            argmin = model._debug['argmin']

            old_argmin = np.copy(argmin)
            found_all_feas = True
            for i in range(input_k_shot):
              cur_fea = label_feas_info['name2class2feas'][filenames[i]][target_class]
              diff = np.abs(supportset_np[i] - cur_fea[np.newaxis]).sum(axis=-1)[:,0,0]
              selected_ind = diff.argmin()
              if diff[selected_ind]<1e-12:
                selected_label = np.where(pinds[i] == selected_ind)[0][0]
                old_argmin[i] = selected_label
              else:
                found_all_feas = False

            energy = gm.evaluate(argmin)
            old_energy = gm.evaluate(old_argmin)
            delta_energy = energy - old_energy
            print(supportset_np.shape, found_all_feas, energy, old_energy, delta_energy)
            if found_all_feas and delta_energy > 0:
              print("WARNING>>> {}".format(batch_index))
            ####


          for i in range(input_k_shot):
            selected_ind = pinds[i][selected_labels[i]]
            selected_fea = supportset_np[i, selected_ind]
            label_feas_info['name2class2feas'][filenames[i]][target_class] = selected_fea
        ##
        selected_inds = res['scores'] == 1
        not_selected  = res['scores'] == 0
        # NOTE: 
        # we use the highest score at the end. for bcd we should use the last element for each
        # class. If we use energy it is hard to rewrite to result_list in ../result_list_utils.py
        res['scores'][selected_inds] = -energy if 'bcd' not in FLAGS.mrf_type else batch_index
        res['scores'][not_selected]  = -1.e8 #small number
        result_dict[mrf_name] = res
        meta_info[mrf_name+'_inference_time'] = [inf_time]
        meta_info[mrf_name+'_energy'] = [energy]
      for key in result_dict.keys():
        if key.startswith('Tree'):
          result_dict.pop(key)
      meta_info.pop('k2_cross_similarity_scores', None)
      meta_info.pop('k2_cross_similarity_pairs', None)
      meta_info.pop('bcd_pairwises', None)
    localtime = time.localtime()
    #print("[{},{},{}]: INFERENCE DONE".format(localtime.tm_hour, localtime.tm_min, localtime.tm_sec))

    result_dict['groundtruth']['target_class'] = result_dict.pop('target_class')
    det_evaluator._postprocess_result_dict(result_dict)
    meta_info['forward_time'] = [forward_time]
    counters['success'] += 1
    global_step = tf.train.global_step(sess, slim.get_global_step())
    if batch_index < eval_config.num_visualizations:
        eval_util.visualize_detection_results(
            result_dict, batch_index, global_step,
            summary_dir=eval_dir,
            export_dir=eval_config.visualization_export_dir,
            show_groundtruth=True,
            max_num_predictions=10,
            min_score_thresh=-10000.0,
            model_name=model_name)

    #for key in result_dict.keys():
    #  if 'multiclass' in key:
    #    result_dict.pop(key)
    result_dict['meta_info'] = meta_info
    if required_batch_indices:
      dummy_result_dict = result_dict
      if batch_index in required_batch_indices:
        res = dict(result_dict)
        res.pop('original_image', None)
        res.pop('meta_info', None)
        res.pop('groundtruth', None)
        saved_results.append(res)
        name = res.keys()[0]
        corloc1 = eval_util.corloc(result_dict[name])[0]
        condition = corloc1 == 6/8.0
        if condition:
          print('Good: id {} corloc: {}'.format(batch_index, corloc1))
        if batch_index == required_batch_indices[-1]:
          name = res.keys()[0]
          import pickle
          with open('{}.pkl'.format(name), 'wb') as f:
            pickle.dump([saved_results, required_batch_indices], f)
    result_dict['meta_info']['evaluated_k2_scores_ratio'] = [np.mean(
      result_dict['meta_info']['evaluated_k2_scores_ratio'])]
    result_dict['meta_info'].pop('energy')
    result_dict['groundtruth'].pop('negative_classes')
    result_dict.pop('original_image', None)
    return result_dict

  def _process_aggregated_results(result_lists):
    eval_metric_fn_key = eval_config.metrics_set
    if eval_metric_fn_key not in EVAL_METRICS_FN_DICT:
      raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
    return det_evaluator.EVAL_METRICS_FN_DICT[eval_metric_fn_key](result_lists,
                                                         categories=categories)

  eval_util.repeated_checkpoint_run(model_names=input_config_names,
      aggregation_params=eval_config.aggregation_params,
      extract_prediction_tensors_fn_list=extract_prediction_tensors_fn_list,
      summary_dir=eval_dir,
      override_elems='bcd' in FLAGS.mrf_type,
      aggregated_result_processor=_process_aggregated_results,
      batch_processor=_process_batch,
      checkpoint_dirs=[checkpoint_dir],
      variables_to_restore=None,
      num_batches=eval_config.num_examples,
      eval_interval_secs=eval_config.eval_interval_secs,
      max_number_of_evaluations=1,
      master=eval_config.eval_master,
      save_graph=eval_config.save_graph,
      save_graph_dir=(eval_dir if eval_config.save_graph else ''),
      use_moving_averages=eval_config.use_moving_averages,
      load_from_vanilla_into_doubled_net=eval_config.load_from_vanilla_into_doubled_net,
      transfered_network_checkpoint=eval_config.transfered_network_checkpoint)
