import re
import json
from . import Collection, Tensor
from .memory import CSCMemoryLayout

import importlib.util
lxmlSpec = importlib.util.find_spec('lxml')
etreeSpec = importlib.util.find_spec('lxml.etree') if lxmlSpec else None
if etreeSpec:
  etree = etreeSpec.loader.load_module()
else:
  etree = importlib.util.find_spec('xml.etree.ElementTree').loader.load_module()

def __createCollection(matrices):
  maxIndex = dict()
  legalName = re.compile('^([a-zA-Z0-9]+)(\[([0-9]+)\])?$')
  for name, matrix in matrices.items():
    match = legalName.match(name)
    if match == None:
      raise ValueError('Illegal matrix name', name, 'in', xmlFile)
    if match.lastindex == 2:
      baseName = match.group(1)
      maxIndex[baseName] = max(maxIndex.get(baseName, 0), int(match.group(3)))
  
  collection = Collection()
  simpleKeys = matrices.keys() - maxIndex.keys()
  for key in simpleKeys:
    collection.__dict__[key] = matrices[key]
  
  for key, value in maxIndex.items():
    collection.__dict__[key] = [ matrices.get('{}[{}]'.format(key, i), None) for i in range(value+1)]

  return collection

def __processMatrix(name, rows, columns, entries, clones, transpose, alignStride):  
  if transpose:
    rows, columns = columns, rows

  matrix = dict()
  for entry in entries:
    row = int(entry[0])-1
    col = int(entry[1])-1
    if transpose:
      matrix[(col, row)] = entry[2]
    else:
      matrix[(row, col)] = entry[2]

  matrices = dict()
  if name in clones:
    for clone in clones[name]:
      matrices[clone] = Tensor(clone, (rows, columns), matrix, alignStride=alignStride)
  else:
    matrices[name] = Tensor(name, (rows, columns), matrix, alignStride=alignStride)
  return matrices

def __complain(child):
  raise ValueError('Unknown tag ' + child.tag)

def parseXMLMatrixFile(xmlFile, clones=dict(), transpose=False, alignStride=None):
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

      matrices.update( __processMatrix(name, rows, columns, entries, clones, transpose, alignStride) )
    else:
      __complain(node)

  return __createCollection(matrices)

def parseJSONMatrixFile(jsonFile, clones=dict(), transpose=False, alignStride=None):
  matrices = dict()

  with open(jsonFile) as j:
    content = json.load(j)
    for m in content:
      entries = m['entries']
      if len(next(iter(entries))) == 2:
        entries = [(entry[0], entry[1], True) for entry in entries]
      matrices.update( __processMatrix(m['name'], m['rows'], m['columns'], entries, clones, transpose, alignStride) )

  return __createCollection(matrices)

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
        if matrixName not in db:
          raise ValueError('Unrecognized matrix name ' + matrixName)
        if len(groups[groupName]) > 0:
          lastMatrixInGroup = groups[groupName][-1]
          if db[lastMatrixInGroup].shape() != db[matrixName].shape():
            raise ValueError('Matrix {} cannot be in the same group as matrix {} due to different shapes.'.format(matrixName, lastMatrixInGroup))
        groups[groupName].append( matrixName )
      else:
        __complain(group)
    # equalize sparsity pattern
    if not noMutualSparsityPattern:
      spp = None
      for matrix in groups[groupName]:
        spp = spp + db[matrix].spp() if spp is not None else db[matrix].spp()
      for matrix in groups[groupName]:
        db[matrix].updateSparsityPattern(spp)

  for matrix in root.findall('matrix'):
    group = matrix.get('group')
    name = matrix.get('name')
    sparse = matrix.get('sparse', '').lower() in strtobool

    if group in groups or name in clones or name in db:
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
        if sparse:
          db[n].setMemoryLayout(CSCMemoryLayout)
        else:
          db[n].setMemoryLayout(DenseMemoryLayout, alignStride=db[n].memoryLayout().alignedStride())
    else:
      raise ValueError('Unrecognized matrix name ' + name)
