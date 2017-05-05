import numpy as np
from scipy.sparse import coo_matrix


def readtrain_createcsr():
  train_file = 'tiny.dat'
  row_id = 0
  rows = []
  cols = []
  vals = []
  max_feature = 0
  with open(train_file) as f:
    for line in f:
      doc = line.split()
      # print(doc)
      features = len(doc)
      for i in xrange(0, features, 2):
        # print('{} ) {} - {} :: {} - {}'.format(row_id, i, i+1, doc[i], doc[i+1]))
        c = int(doc[i])
        v = int(doc[i+1])
        if(c > max_feature):
          max_feature = c
        rows.append(row_id)
        cols.append(c)
        vals.append(v)
      row_id += 1
  # print('row_id : {}'.format(row_id))
  # print('max_feature : {}'.format(max_feature))
  # print('rows : {}'.format(rows))
  # print('cols : {}'.format(cols))
  # print('vals : {}'.format(vals))
  rows = np.array(rows)
  cols = np.array(cols)
  vals = np.array(vals)
  max_feature += 1
  return coo_matrix((vals, (rows, cols)), shape=(row_id, max_feature )).tocsr()

def main():
  train_csr = readtrain_createcsr()
  print(train_csr)

if __name__ == '__main__':
  main()