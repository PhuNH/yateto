#!/usr/bin/env python3

from yateto import *
from yateto.gemm_configuration import *

def gemm_cfg(arch, variant):
  if variant and variant != 'polly':
    concrete_gemm = getattr(gemm_configuration, variant)
    return GeneratorCollection([concrete_gemm(arch)])
  else:
    return None

_bench_no = 0
def add_tensor(name, ind, size):
  shape = tuple(size[k] for k in ind)
  return Tensor(name + str(_bench_no), shape)

def add_bench(g, descr, sizes):
  global _bench_no

  Qind, tmp_ind, Find = descr.split('-')
  size = {k: int(s) for k,s in re.findall(r'([a-z]):([0-9]+)', sizes)}

  Q = add_tensor('Q', Qind, size)
  tmp = add_tensor('tmp', tmp_ind, size)
  F = add_tensor('F', Find, size)

  g.add(sizes.translate(str.maketrans(':;', '__')), Q[Qind] <= Q[Qind] + F[Find] * tmp[tmp_ind])
  _bench_no = _bench_no + 1

def add(g):
  add_bench(g, 'xyzp-xyp-z', 'x:16;y:16;z:16;p:4')
