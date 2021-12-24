#!/usr/bin/env python3

from yateto import *
from yateto.gemm_configuration import *

def gemm_cfg(arch, variant):
  concrete_gemm = getattr(gemm_configuration, variant)
  return GeneratorCollection([concrete_gemm(arch)])

def add(g):
  M = 32
  N = 32
  K = 32
  A = Tensor('A', (M, K))
  B = Tensor('B', (K, N))
  C = Tensor('C', (M, N))

  g.add('matmulAB', C['ij'] <= A['ik'] * B['kj'])
