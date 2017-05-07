import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances

def readtrain_createcsr():
  # train_file = 'tiny.dat'
  train_file = 'train.dat'
  row_id = 0
  rows = []
  cols = []
  vals = []
  max_feature = 0
  with open(train_file) as f:
    for line in f:
      doc = line.split()
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
  rows = np.array(rows)
  cols = np.array(cols)
  vals = np.array(vals)
  max_feature += 1
  print('Data Read Succesfully...')
  return coo_matrix((vals, (rows, cols)), shape=(row_id, max_feature )).tocsr()



def K_Means(csr_data, K = 3, max_epoch = 10, max_iterations = 10):
  print('Starting K Means')
  print('K : {}, max_Epoch : {}, max_Iterations : {} \n'.format(K, max_epoch, max_iterations))
  docs = 8580
  cluster = {}
  f = open('res_K_Means.txt', 'w')
  f.write('K : {}, max_Epoch : {}, max_Iterations : {} \n'.format(K, max_epoch, max_iterations))
  old_cluster = cluster
  e = 1
  while True:
    print('{} Epoch : {} {} '.format('-'*10, e, '-'*10))
    f.write('{} Epoch : {} {} \n'.format('-'*10, e, '-'*10))
    centroid_indices = np.random.choice(docs, K, replace=False)
    centroid_vectors = csr_data[centroid_indices].toarray()

    # Loop Begins
    for itr in range(max_iterations):
      cluster = {}
      for i in range(K):
        cluster[i] = []

      cv = csr_matrix(centroid_vectors)
      dist = pairwise_distances(csr_data, cv) # dist between data & cluster centroids
      # axis 0 for column, return the index of the min dist for that column
      # axis 1 for row, this is what we need to tell which row has shortest dist to which cluster
      min_dist = dist.argmin(axis=1)
      for i in range(len(min_dist)):
        cluster[min_dist[i]].append(i) # assiging the vect (rowID) to closest cluster

      # readjusting the centroids by taking the mean
      for i in range(K):
        clust_vect_indices = cluster[i]
        clust_vect_indices_data = csr_data[clust_vect_indices].toarray()
        clust_mean = np.mean(clust_vect_indices_data, axis=0)
        centroid_vectors[i] = clust_mean
      print('Iteration : {}'.format(itr))
      if(old_cluster == cluster): # no change detected
        print('-----------NOCHNAGE termination EPoch--------')
        break
      old_cluster = cluster

    print('Epoch {} completed '.format(e))
    # print('\n~~~~ Epoch : {} Cluster : {}\n'.format(e, cluster))
    f.write('\n~~~~ Epoch : {} Cluster : {}\n\n'.format(e, cluster))
    write_Cluster(cluster, e)

    e += 1 # either we have reached the max_EPoCh
    if(e == max_epoch):
      break
      # Loop Ends

  print('Fianl cluster assignments Writen')
  f.write('Fianl cluster assignments : {}\n'.format(cluster))
  f.close()
  print('Sending to write assiginments')
  write_Cluster(cluster,e)

import os.path
from datetime import datetime
def write_Cluster(cld, e):
  subdir = 'results'+str(datetime.now()) + '_epcoh_{}_res'.format(e)
  print(subdir)
  try:
    os.mkdir(subdir)
  except Exception:
    print(Exception)
  f = open(os.path.join(subdir, 'KM_res_assig_{}.txt'.format(e) ), 'w')
  resD = {}
  for k in cld:
    for v in cld[k]:
      resD[v] = k
  for k in sorted(resD.keys()):
    f.write('{}\n'.format(resD[k] + 1))
  f.close()

def main_two():
  train_csr = readtrain_createcsr()
  K = 7
  max_epoch = 10
  max_iterations = 100
  K_Means(train_csr, K, max_epoch, max_iterations)

if __name__ == '__main__':
  main_two()
