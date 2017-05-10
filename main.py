import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
import os.path
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD


# a utility method to write the results of the cluster to file
def write_Cluster(cld, e):
  subdir = 'results' + str(datetime.now()) + '_epcoh_{}_res'.format(e)
  print(subdir)
  try:
    os.mkdir(subdir)
  except Exception:
    print(Exception)
  f = open(os.path.join(subdir, 'KM_res_assig_{}.txt'.format(e)), 'w')
  resD = {}
  for k in cld:
    for v in cld[k]:
      resD[v] = k
  for k in sorted(resD.keys()):
    f.write('{}\n'.format(resD[k] + 1))
  f.close()

# the method to read the input train data and create a CSR Matrix to return
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

#This method uses PairwiseDistance as metric for determining the closeness of any vectors to the centroids

def K_Means_PairWise(csr_data, K = 3, max_epoch = 10, max_iterations = 10):
  print('Starting K Means Pairwise Metric')
  print('\t [ K : {}, max_Epoch : {}, max_Iterations : {} ]\n'.format(K, max_epoch, max_iterations))
  # docs = 5
  docs = 8580
  cluster = {}
  f = open('res_K_Means.txt', 'w')
  f.write('K : {}, max_Epoch : {}, max_Iterations : {} \n'.format(K, max_epoch, max_iterations))
  old_cluster = cluster # a reference to the old_cluster, this is particularly used to check if this cluster is completly
  # similar to the newly found cluster if it is so, we terminate the epoch early
  e = 1 # the epoch variable
  # the outerloop for epochs
  while True:
    print('{} Epoch : {} {} '.format('-'*10, e, '-'*10))
    f.write('{} Epoch : {} {} \n'.format('-'*10, e, '-'*10))

    centroid_indices = np.random.choice(docs, K, replace=False) # as per KMeans algorithm we first select K random points
    # as our centroids, on the above line we are doing same by selecting any random K vectors from the dataset without
    # replacement , so that we dont get two centroids as same
    centroid_vectors = csr_data[centroid_indices].toarray() # getting dense version of the centtoids

    # Loop Begins
    for itr in range(max_iterations):
      cluster = {}
      for i in range(K):
        cluster[i] = [] # creating empty clusters before we begin

      cv = csr_matrix(centroid_vectors) # getting the csr version back from centroids
      dist = pairwise_distances(csr_data, cv) # calculating distance between data & cluster centroids
      # axis 0 for column, return the index of the min dist for that column
      # axis 1 for row, this is what we need to tell which row has shortest dist to which cluster
      min_dist = dist.argmin(axis=1)  # as we are using dist as the metric for closeness we will select the minimum distance, means the closeset
      for i in range(len(min_dist)):
        cluster[min_dist[i]].append(i) # assiging the vect (rowID) to closest cluster

      # readjusting the centroids by taking the mean
      for i in range(K):
        clust_vect_indices = cluster[i]
        clust_vect_indices_data = csr_data[clust_vect_indices].toarray()
        clust_mean = np.mean(clust_vect_indices_data, axis=0)
        centroid_vectors[i] = clust_mean  # changing the centroid of the cluster with its new mean
      print('Iteration : {}'.format(itr))
      if(old_cluster == cluster): # if no change detected, then terminate
        print('-----------NOCHNAGE termination EPoch--------')
        break
      old_cluster = cluster # assign this cluster to old for comparision in the next iteration

    print('Epoch {} completed '.format(e))
    # print('\n~~~~ Epoch : {} Cluster : {}\n'.format(e, cluster))
    # f.write('\n~~~~ Epoch : {} Cluster : {}\n\n'.format(e, cluster))
    for k in sorted(cluster.keys()):
      f.write('\ncluster : {} : size : {} \n'.format(k, len(cluster[k])))
    write_Cluster(cluster, e)
    # calculate_SumofSquareDistanceBySize(cluster, csr_data, e) A util method to calculate score

    e += 1 # either we have reached the max_EPoCh
    if(e == max_epoch):
      break
      # Loop Ends

  print('Fianl cluster assignments Writen')
  f.write('Fianl cluster assignments : {}\n'.format(cluster))
  f.close()
  print('Sending to write assiginments')
  write_Cluster(cluster,e)  # write the labels of the cluster in the files


# in this method we are using cosine similarity as the metric to determine the closeness among the points
def K_Means_CosineSimilarity(csr_data, K = 3, max_epoch = 10, max_iterations = 10):
  print('Starting K Means cosine Similaity Metric ')
  print('\t [ K : {}, max_Epoch : {}, max_Iterations : {} ]\n'.format(K, max_epoch, max_iterations))
  # docs = 5
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
      cos_sim =  cosine_similarity(csr_data, cv, dense_output=False)# similarity between data & cluster centroids
      # axis 1 for row, this is what we need to tell which row has shortest dist to which cluster
      max_sim = cos_sim.argmax(axis=1).getA1()  # getA1() for flattening the matrix
      # as we are using consine similarity for the closeness metric, we have to select which is having the most similarity
      # to us, thus we are selecting the max here.

      for iv in range(max_sim.shape[0]):
        cluster[max_sim[iv]].append(iv)# assiging the vect (rowID) to closest cluster

      # readjusting the centroids by taking the mean
      for i in range(K):
        clust_vect_indices = cluster[i]
        clust_vect_indices_data = csr_data[clust_vect_indices].toarray()
        clust_mean = np.mean(clust_vect_indices_data, axis=0)
        centroid_vectors[i] = clust_mean  # assinging the new mean of the cluster as the new centroid of the cluster
      print('Iteration : {}'.format(itr))


      if(old_cluster == cluster): # if no change detected in the cluster from prevouos iteration then terminate
        print('-----------NOCHNAGE termination EPoch--------')
        break
      old_cluster = cluster

    print('Epoch {} completed '.format(e))
    # print('\n~~~~ Epoch : {} Cluster : {}\n'.format(e, cluster))
    # f.write('\n~~~~ Epoch : {} Cluster : {}\n\n'.format(e, cluster))
    for k in sorted(cluster.keys()):
      f.write('\ncluster : {} : size : {} \n'.format(k, len(cluster[k])))
    write_Cluster(cluster, e)
    # calculate_SumofSquareDistanceBySize(cluster, csr_data, e)

    e += 1 # either we have reached the max_EPoCh
    if(e == max_epoch):
      break
      # Loop Ends

  print('Fianl cluster assignments Writen')
  f.write('Fianl cluster assignments : {}\n'.format(cluster))
  f.close()
  print('Sending to write assiginments')
  write_Cluster(cluster,e)

# A utility method to calculate the Sum of Squared Distance of the points in the cluster divided by the cluster size
def calculate_SumofSquareDistanceBySize(cluster, csr_data, epoch):
  print('-' * 50)
  c_dist = 0
  try:
    f = open('ssd_res_{}.txt'.format(epoch),'w')
    f.write('{} Epcoch : {}\n'.format('-' * 20, epoch, '-' * 20))
    for i in cluster:
      cl = cluster[i]
      clv = csr_data[cl].toarray()
      center = np.mean(clv, 0)
      center_sp = csr_matrix(center)
      cl_data = csr_data[cl]
      dist = pairwise_distances(cl_data, center_sp)  # dist between data & cluster centroids
      dist_sum = np.sum(dist)
      c_dist += dist_sum / len(cluster[i])
      print('Cluster # : {}, Size :{},  Sum of Squared Distances : {}\n'.format(i, len(cluster[i]), dist_sum))
      f.write('Cluster # : {}, Size :{},  Sum of Squared Distances : {}\n'.format(i, len(cluster[i]), dist_sum))

  except Exception:
    print('seom except')

  print('Epoch # : {} , Sum of clusterSSD / clusterSize : {}\n'.format(epoch, c_dist))
  print('-' * 50)
  f.write('Epoch  # : {} , Sum of clusterSSD / clusterSize : {}\n'.format(epoch, c_dist))
  f.close()

def main():
  train_csr = readtrain_createcsr()
  train_csr = normalize(train_csr, norm='l2', axis=1)

  # tried dimension reduction but resulted in decreased scores
  # dr_comp = 2
  # dr_comp = 1200
  # print('\nreducing dimension from {} to {}'.format(train_csr.shape, dr_comp))
  # svd = TruncatedSVD(n_components=dr_comp, n_iter=7, random_state=42)
  # train_Dr = svd.fit_transform(train_csr)
  # print('\nnew reduced dimension {}'.format(train_Dr.shape))
  # train_csr_Dr = csr_matrix(train_Dr)
  # print('type before : {}'.format(type(train_csr)))
  # print('type after : {}'.format(type(train_csr_Dr)))
  # DR ends

  K = 7
  max_epoch = 10
  max_iterations = 100
  K_Means_PairWise(train_csr, K, max_epoch, max_iterations)
  # K_Means_CosineSimilarity(train_csr, K, max_epoch, max_iterations)

if __name__ == '__main__':
  main()




