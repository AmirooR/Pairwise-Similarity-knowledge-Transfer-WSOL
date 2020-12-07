from mrf_evaluation import get_extra_dense_indices
from greedy import GreedyCoObj
import numpy as np
from IPython import embed

cobjs = [GreedyCoObj(vars=[i], values=[[0],[1]]) for i in range(4)]
jco = cobjs[0].join(cobjs[1])
jco2 = cobjs[2].join(cobjs[3])
co = jco.join(jco2)
factors = {}

for i in range(4):
    factors[i] = np.random.random_sample(2,)

indices = [(i,i+1) for i in range(0,4,2)]
extra_indices = get_extra_dense_indices(4)
indices.extend(extra_indices)

for e in indices:
    factors[e] = np.random.random_sample((2,2))

evaluations = co.evaluate(factors, 10)
embed()
