import importlib.util
gb_spec = importlib.util.find_spec('chainforge')
try:
  if gb_spec:
    gb = gb_spec.loader.load_module()
    from .external_generator import FusedGemms
except:
  raise ('Found chainforge spec but cannot load. Please, check installation of chainforge')


class Description(object):
  def __init__(self, node, result, arguments, add, scalar):
    self.node = node
    self.result = result
    self.args = arguments
    self.add = add
    self.scalar = scalar
    self._inter_counter: int = 0

  def __iter__(self):
    self._inter_counter = 0
    return self

  def __next__(self):
    index = self._inter_counter
    args_index = 3 * index
    self._inter_counter += 1
    try:
      return (self.node.get_child(index),
              self.args[args_index:args_index + 3],
              self.add[index],
              self.scalar[index])
    except IndexError:
      raise StopIteration


def generator(arch, descr, target):
  if target == 'gpu' and gb_spec:
    return FusedGemms(arch, descr)
  else:
    raise NotImplementedError(f'no implementation found for {target} target')
