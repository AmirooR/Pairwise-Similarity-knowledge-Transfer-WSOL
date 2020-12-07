import itertools
import numpy as np

class GreedyCoObj:
  def __init__(self, vars, values):
    self.vars = vars
    self.values = values

  def join(self, cobj):
    vars = self.vars + cobj.vars
    values = []
    for x, y in itertools.product(self.values, cobj.values):
      values.append(x+y)
    return GreedyCoObj(vars=vars, values=values)

  def evaluate(self, factors, num_to_keep, use_scores=True):
    '''
    params:
      use_scores:
        use score and multiply them if true. otherwise, uses logits log(s/(1-s))
        where s is the score and adds them
    '''
    vars_set = set(self.vars)
    evaluations = []
    for value in self.values:
      prob = 1.0 if use_scores else 0.0
      for f_vars, f_potential in factors.items():
        subset = set()
        i_vars = f_vars
        if type(f_vars) == int:
          subset.add(f_vars)
          i_vars = [f_vars]
        else:
          subset = set(f_vars)
        if subset.issubset(vars_set): # all of the factor vars are in the list
          ret = f_potential.copy()
          for v in i_vars:
            ret = ret[value[self.vars.index(v)]]
          if use_scores:
            prob *= ret
          else:
            a = 0.09
            prob += ret**a / (ret**a + (1-ret)**a)
            # logit np.log(ret/(1-ret))
      evaluations.append(prob)
    evaluations = np.array(evaluations)
    sort_idx = np.argsort(-evaluations)
    self.values = [self.values[i] for i in sort_idx]
    if num_to_keep < len(self.values):
      self.values = self.values[:num_to_keep]
    return evaluations[sort_idx]
