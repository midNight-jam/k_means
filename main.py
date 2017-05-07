import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
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

def centroid(data):
  """Find the centroid of the given data."""
  print('Centroid : {}'.format(data))
  return np.mean(data, 0) # 0: means of columns


def sse(data):
  """Calculate the SSE of the given data."""
  if (data.ndim == 1):
    # z = np.zeros(len(data))
    z = data[:]
    data = np.vstack((data, z))
    # print('Combined')
  u = centroid(data)
  return np.sum(np.linalg.norm(x = data - u, ord=2, axis=1))

  # print('-'*50)
  # print('data shape : {}'.format(data.shape))
  # print('type ofdata : {}'.format(type(data)))
  # print('data : {}'.format(data))
  # print('type data : {}'.format(type(data)))
  # print('u shape : {}'.format(u.shape))
  # print('u : {}'.format(u))
  # print('type u : {}'.format(type(u)))

def KMeans(mat, k, epoch, iterations):
  for j in range(1):
    print(j)
    ind = np.random.choice(len(mat), k, replace=False) # choose K random indexes
    # print('ind : {}'.format(ind))
    u = mat[ind,:]  # the centroids chosen randomly
    # print(' First u   : {}'.format(u))
    # print('red mat : {}'.format(mat))
    t = 0
    sum_squared_errors = np.inf

    while True:
      t += 1

      # Cluster assignment
      C = [None] * k
      for x in mat:
        j = np.argmin(np.linalg.norm(x - u, 2, 1))
        C[j] = x if C[j] is None else np.vstack((C[j], x))

      # print('C : {}'.format(C))
      # Centroid update
      for j in range(k):
        u[j] = centroid(C[j])

      new_sse = np.sum([sse(C[j]) for j in range(k)])
      # print('NSEE : {}'.format(new_sse))
      gain = sum_squared_errors - new_sse
      # print('gain : {}'.format(gain))

      if(k==iterations):
        break
  return C, u


def main():
  train_csr = readtrain_createcsr()
  mat = train_csr.toarray()
  print(mat.shape)
  print(mat)
  k = 2
  epoch = 1
  iters = 2
  print('Starting Calculation')
  clusters, centroids = KMeans(mat, k, epoch, iters)
  print('completed Calculation')

  # print('clusters.shape : {} '.format(clusters))
  # print('centroids.shape : {} '.format(centroids))

  f = open('resKMeans.txt', 'w')
  for i in range(k):
    # print('Cluster : {} '.format(i))
    f.write('\nCluster : {} \n'.format(i))
    # print('Centroid : {} \n'.format(centroids[i]))
    f.write('Centroid : {} \n'.format(centroids[i]))
    for j in range(len(clusters[i])):
      f.write('        p : {}'.format(clusters[i][j]))
      # print('        p : {}'.format(clusters[i][j]))
      # print('-'*50)


def K_Means(csr_data, K = 3, max_epoch = 10, max_iterations = 10):
  print('Starting K Means')
  print('K : {}, max_Epoch : {}, max_Iterations : {} \n'.format(K, max_epoch, max_iterations))
  docs = 8580
  # docs = 5
  cluster = {}
  f = open('res_K_Means.txt', 'w')
  f.write('K : {}, max_Epoch : {}, max_Iterations : {} \n'.format(K, max_epoch, max_iterations))
  old_cluster = cluster
  e = 1
  # for e in range(max_epoch):
  while True:
    print('{} Epoch : {} {} '.format('-'*10, e, '-'*10))
    f.write('{} Epoch : {} {} \n'.format('-'*10, e, '-'*10))
    centroid_indices = np.random.choice(docs, K, replace=False)
    # print('Centroids : {}'.format(centroid_indices))
    centroid_vectors = csr_data[centroid_indices].toarray()
    # print('vectors : {}'.format(centroid_vectors))

    # Loop Begins
    for itr in range(max_iterations):
      cluster = {}
      for i in range(K):
        cluster[i] = []

      cv = csr_matrix(centroid_vectors)
      dist = pairwise_distances(csr_data, cv) # dist between data & cluster centroids
      # print(dist)
      # axis 0 for column, return the index of the min dist for that column
      # axis 1 for row, this is what we need to tell which row has shortest dist to which cluster
      min_dist = dist.argmin(axis=1)
      # print(min_dist)
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

    # cmean = np.mean(centroid_vectors, axis=0)
    # print('cmean : {}'.format(cmean))
    # break
    e += 1 # either we have reached the max_EPoCh
    if(e == max_epoch):
      break
      # Loop Ends

  print('Fianl cluster assignments Writen')
  f.write('Fianl cluster assignments : {}\n'.format(cluster))
  f.close()
  print('Sending to write assiginments')
  write_Cluster(cluster,e)

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
  # print(train_csr.toarray())
  mean  = train_csr.mean(axis=0)
  # print('means : {}'.format(mean))
  lin_norm = norm(train_csr)
  # print('nrom : {}'.format(lin_norm))
  distances = pairwise_distances(train_csr, train_csr)
  # print(distances)
  K = 7
  max_epoch = 10
  max_iterations = 50
  K_Means(train_csr, K, max_epoch, max_iterations)

def tryD():
  cld = {}
  resD = {}
  cld[0] = [1, 4, 7, 10, 13]
  cld[1] = [2, 5, 8, 11, 14]
  cld[2] = [3, 6, 9, 12, 15]

  resD[0] = [1, 4, 7, 10, 13]
  resD[1] = [2, 5, 8, 11, 14]
  resD[2] = [9, 6, 9, 12, 15]

  print('Eual : {}'.format(cld == resD))
  # write_Cluster(cld)

import os.path
from datetime import datetime
def tryF():
  subdir = str(datetime.now()) + '-res'
  print(subdir)
  try:
    os.mkdir(subdir)
  except Exception:
    print(Exception)
  f = open(os.path.join(subdir,'res__'+'.txt'), 'w')
  f.write('Some shit')
  f.close()


if __name__ == '__main__':
  # main()
  main_two()
  # tryD()
  # tryF()
