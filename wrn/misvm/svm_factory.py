import misvm
from functools import partial
import itertools

def get_classifiers(classifier_params):
  classifiers = {}
  for svm_type, svm_param_dict in classifier_params.iteritems():
    parameter_names = svm_param_dict.keys()
    parameter_value_lists = svm_param_dict.values()
    value_combinations = itertools.product(*parameter_value_lists)
    for values in value_combinations:
      classifier_name = svm_type
      kwargs = {}
      for i, val in enumerate(values):
        print('{}:{}'.format(parameter_names[i], val))
        classifier_name += '_{}{}'.format(parameter_names[i], val)
        kwargs[parameter_names[i]] = val
      if svm_type == 'sbMIL':
        svm_cls = misvm.sbMIL
      elif svm_type == 'miSVM':
        svm_cls = misvm.miSVM
      elif svm_type == 'MISVM':
        svm_cls = misvm.MISVM
      else:
        raise ValueError('SVM type {} is not supported'.format(svm_type))
      classifiers[classifier_name] = partial(svm_cls, **kwargs)
  return classifiers
