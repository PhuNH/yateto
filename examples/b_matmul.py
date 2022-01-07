#!/usr/bin/env python3

from yateto import *
from yateto.gemm_configuration import *

def gemm_cfg(arch, variant):
  if variant:
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

  Cind, Aind, Bind = descr.split('-')
  size = {k: int(s) for k,s in re.findall(r'([a-z]):([0-9]+)', sizes)}

  A = add_tensor('A', Aind, size)
  B = add_tensor('B', Bind, size)
  C = add_tensor('C', Cind, size)

  g.add(sizes.translate(str.maketrans(':;', '__')), C[Cind] <= A[Aind] * B[Bind])
  _bench_no = _bench_no + 1

def add(g):
  add_bench(g, 'ij-ik-kj', 'i:32;j:32;k:32')
  add_bench(g, 'ij-ik-kj', 'i:64;j:12;k:64')
  add_bench(g, 'ij-ik-kj', 'i:64;j:32;k:64')
  add_bench(g, 'ij-ik-kj', 'i:64;j:64;k:64')
  add_bench(g, 'ij-ik-kj', 'i:128;j:128;k:128')
  add_bench(g, 'ij-ik-kj', 'i:256;j:256;k:256')
  add_bench(g, 'ij-ik-kj', 'i:512;j:512;k:512')
