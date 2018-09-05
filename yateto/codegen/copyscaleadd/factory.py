from ..common import *
from .generic import Generic

class Description(object):
  def __init__(self, alpha, beta, result: IndexedTensorDescription, term: IndexedTensorDescription):
    self.alpha = alpha
    self.beta = beta
    self.result = result
    self.term = term
    
    assert self.beta == 1.0 or self.beta == 0.0, 'copyscaleadd supports only beta=0.0 or beta=1.0 at the moment.'
    
    assert self.result.indices == self.term.indices
    rA = loopRanges(self.term, self.term.indices)
    rB = loopRanges(self.result, self.result.indices)
    assert testLoopRangesAContainedInB(rA, rB)
    
    self.loopRanges = rA
    

def generator(arch, descr):
  return Generic(arch, descr)
