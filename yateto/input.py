import re
import itertools
import json
from . import Collection, Tensor
from .memory import CSCMemoryLayout, DenseMemoryLayout
from . import aspp
from .util import create_collection

import importlib.util
lxmlSpec = importlib.util.find_spec('lxml')
etreeSpec = importlib.util.find_spec('lxml.etree') if lxmlSpec else None
if etreeSpec:
  etree = etreeSpec.loader.load_module()
else:
  etree = importlib.util.find_spec('xml.etree.ElementTree').loader.load_module()

def __transposeMatrix(matrix):
  matrixT = dict()
  for entry,value in matrix.items():
    matrixT[(entry[1], entry[0])] = value
  return matrixT

def __processMatrix(name, rows, columns, entries, clones, transpose, alignStride, namespace=None):
  matrix = dict()
  for entry in entries:
    row = int(entry[0])-1
    col = int(entry[1])-1
    matrix[(row, col)] = entry[2]

  matrices = dict()
  names = clones[name] if name in clones else [name]
  for name in names:
    shape = (columns, rows) if transpose(name) else (rows, columns)
    if shape[1] == 1:
      shape = (shape[0],)
    mtx = __transposeMatrix(matrix) if transpose(name) else matrix
    if len(shape) == 1:
      mtx = {(i[0],): val for i,val in mtx.items()}
    matrices[name] = Tensor(name, shape, mtx, alignStride=alignStride(name), namespace=namespace)
  return matrices

def __processTensor(name, rank, shape, entries, clones, transpose, alignStride, namespace=None):
  tensor = dict()
  for entry in entries:
    
    index = [0] * rank
    for i in range(rank-1):
      index[i] = entry[i]-1
      
    tensor[tuple(index)] = entry[rank]

  tensors = dict()
  names = clones[name] if name in clones else [name]
  for name in names:
    shape = tuple(shape)
    #shape = (columns, rows) if transpose(name) else (rows, columns)
    #if shape[1] == 1:
    #  shape = (shape[0],)
    #mtx = __transposeMatrix(matrix) if transpose(name) else matrix
    tensors[name] = Tensor(name, shape, tensor, alignStride=alignStride(name), namespace=namespace)
  return tensors


def __complain(child):
  raise ValueError('Unknown tag ' + child.tag)

def parseXMLMatrixFile(xmlFile, clones=dict(), transpose=lambda name: False, alignStride=lambda name: False, namespace=None):
  tree = etree.parse(xmlFile)
  root = tree.getroot()
  
  matrices = dict()
  
  for node in root:
    if node.tag == 'matrix':
      name = node.get('name')
      rows = int( node.get('rows') )
      columns = int( node.get('columns') )

      entries = list()
      for child in node:
        if child.tag == 'entry':
          row = int(child.get('row'))
          col = int(child.get('column'))
          entry = child.get('value', True)
          entries.append((row,col,entry))
        else:
          __complain(child)

      matrices.update( __processMatrix(name, rows, columns, entries, clones, transpose, alignStride, namespace) )
    else:
      __complain(node)

  return create_collection(matrices)

def parseJSONMatrixFile(jsonFile, clones=dict(), transpose=lambda name: False, alignStride=lambda name: False, namespace=None):
  matrices = dict()

  with open(jsonFile) as j:
    content = json.load(j)
    for m in content:
      entries = m['entries']
      if len(next(iter(entries))) == 2:
        entries = [(entry[0], entry[1], True) for entry in entries]
      matrices.update( __processMatrix(m['name'], m['rows'], m['columns'], entries, clones, transpose, alignStride, namespace) )

  return create_collection(matrices)


def parseJSONTensorFile(jsonFile, clones=dict(), transpose=lambda name: False, alignStride=lambda name: False, namespace=None):
  tensors = dict()

  with open(jsonFile) as j:
    content = json.load(j)
    for m in content:
      entries = m['entries']
      if len(next(iter(entries))) == 2:
        entries = [(entry[0], entry[1], True) for entry in entries]
      tensors.update( __processTensor(m['name'], m['rank'], m['shape'], entries, clones, transpose, alignStride, namespace) )

  return create_collection(tensors)


def memoryLayoutFromFile(xmlFile, db, clones):
  tree = etree.parse(xmlFile)
  root = tree.getroot()
  strtobool = ['yes', 'true', '1']
  groups = dict()

  for group in root.findall('group'):
    groupName = group.get('name')
    noMutualSparsityPattern = group.get('noMutualSparsityPattern', '').lower() in strtobool
    groups[groupName] = list()
    for matrix in group:
      if matrix.tag == 'matrix':
        matrixName = matrix.get('name')
        if not db.containsName(matrixName):
          raise ValueError('Unrecognized matrix name ' + matrixName)
        if len(groups[groupName]) > 0:
          lastMatrixInGroup = groups[groupName][-1]
          if db.byName(lastMatrixInGroup).shape() != db.byName(matrixName).shape():
            raise ValueError('Matrix {} cannot be in the same group as matrix {} due to different shapes.'.format(matrixName, lastMatrixInGroup))
        groups[groupName].append( matrixName )
      else:
        __complain(group)
    # equalize sparsity pattern
    if not noMutualSparsityPattern:
      spp = None
      for matrix in groups[groupName]:
        spp = aspp.add(spp, db.byName(matrix).spp()) if spp is not None else db.byName(matrix).spp()
      for matrix in groups[groupName]:
        db.byName(matrix).setGroupSpp(spp)

  for matrix in root.findall('matrix'):
    group = matrix.get('group')
    name = matrix.get('name')
    sparse = matrix.get('sparse', '').lower() in strtobool

    if group in groups or name in clones or db.containsName(name):
      blocks = []
      for block in matrix:
        raise NotImplementedError
        if block.tag == 'block':
          startrow = int(block.get('startrow'))
          stoprow = int(block.get('stoprow'))
          startcol = int(block.get('startcol'))
          stopcol = int(block.get('stopcol'))
          blksparse = (block.get('sparse') == None and sparse) or block.get('sparse', '').lower() in strtobool
        else:
          __complain(block)
      names = groups[group] if group in groups else (clones[name] if name in clones else [name])
      for n in names:
        tensor = db.byName(n)
        if sparse:
          tensor.setMemoryLayout(CSCMemoryLayout)
        else:
          tensor.setMemoryLayout(DenseMemoryLayout, alignStride=tensor.memoryLayout().alignedStride())
    else:
      raise ValueError('Unrecognized matrix name ' + name)
