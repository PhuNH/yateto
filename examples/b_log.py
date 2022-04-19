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
    N = 8
    A = Tensor('A', (N, N))
    B = Tensor('B', (N, N, N))
    w = Tensor('w', (N,))
    C = Tensor('C', (N, N))

    kernel = C['ij'] <= 2.0 * C['ij'] + A['lj'] * B['ikl'] * w['k']
    g.add('kernel', kernel)
