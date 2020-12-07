import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import os, sys

FLAGS = flags.FLAGS

def sigmoid(x):
  return 1.0/(1.0+np.exp(-x))

def add_calibration_functions(is_logit=True):
  '''
  defined flags:
    add_calibration:
    calib_prefix:
    calib_folder:
    add_calib_platt:
    add_calib_isotonic:
    add_calib_histogram:
    calib_nbins:
  '''
  calibration_functions = []
  if FLAGS.add_calibration:
    prefix = FLAGS.calib_prefix
    root_folder = FLAGS.calib_folder
    matches = np.load(os.path.join(root_folder, prefix+'_matches.npy'))
    unmatches = np.load(os.path.join(root_folder, prefix+'_unmatches.npy'))
    if FLAGS.add_calib_platt:
      [a,b] = np.load(os.path.join(root_folder, 'platt_{}.npy'.format(prefix)))
      name_platt = 'calib_platt_{}'.format(prefix)
      g = lambda x: x if is_logit else np.log(x/(1.0-x))
      f_platt = lambda x: (1.0/(1.0 + np.exp(-a*g(x)-b)), name_platt)
      calibration_functions.append(f_platt)
    if FLAGS.add_calib_isotonic:
      from sklearn.isotonic import IsotonicRegression
      nbins = FLAGS.calib_nbins
      m = np.histogram(matches, np.linspace(0,1,nbins+1), normed=FLAGS.calib_histogram_normed)
      u = np.histogram(unmatches, np.linspace(0,1,nbins+1), normed=FLAGS.calib_histogram_normed)
      normalized_ratio = m[0]/(m[0]+u[0]+1e-14)
      Y = normalized_ratio
      X = np.linspace(0,1,nbins+1)[:-1]+0.5/nbins
      ir = IsotonicRegression(y_min=0.0, y_max=1.0)
      X = X[~np.isnan(Y)]
      Y = Y[~np.isnan(Y)]
      ir.fit(X,Y)
      name_ir = 'calib_isotonic_{}'.format(prefix)
      def f_ir(x):
        if is_logit:
          x = sigmoid(x)
        x[x< 0.5/nbins] = 0.5/nbins
        x[x> 1.0 - 0.5/nbins] = 1 - 0.5/nbins
        x_shape = x.shape
        y = ir.predict(x.flatten())
        return y.reshape(x_shape), name_ir
      calibration_functions.append(f_ir)
    if FLAGS.add_calib_histogram:
      nbins = FLAGS.calib_nbins
      m = np.histogram(matches, np.linspace(0,1,nbins+1), normed=FLAGS.calib_histogram_normed)
      u = np.histogram(unmatches, np.linspace(0,1,nbins+1), normed=FLAGS.calib_histogram_normed)
      normalized_ratio_h = m[0]/(m[0]+u[0]+1e-14)
      for i, h in enumerate(normalized_ratio_h):
        last_h = 0.0 if i == 0 else normalized_ratio_h[i-1]
        if np.isnan(h):
          normalized_ratio_h[i] = last_h
      name_hist = 'calib_hist_{}'.format(prefix)
      def f_hist(x):
        if is_logit:
          x = sigmoid(x)
        i_bin = np.int32(x*nbins)
        i_bin = np.clip(i_bin, 0, nbins-1)
        return normalized_ratio_h[i_bin], name_hist
      calibration_functions.append(f_hist)
  return calibration_functions

