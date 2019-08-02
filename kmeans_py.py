import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from copy import deepcopy

# plt.figure(figsize=(12, 12))
#
# n_samples = 3000
# random_state = 3
# n_features = 2
# n_centers = 6
# max_iter = 100
# dist_2_threshold = 0.00001
#
# np.random.seed(random_state)
#
# X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=random_state)
#
# # Incorrect number of clusters
# start_t = time()
# kmeans = KMeans(n_clusters=n_centers, init='random', random_state=random_state).fit(X)
# print("t: {}s".format(time()-start_t))
# print(kmeans.inertia_)
# plt.subplot(121)
# plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
# plt.title("Result of K-Means")



# 计算欧几里得距离
def distEclud(centroids, data_i):
    # return np.sqrt(np.sum(np.power(vecA - vecB, 2)))  # 求两个向量之间的距离
    dist = np.sqrt(np.sum(np.power(centroids - data_i, 2),1))
    # dist_2 = np.sum(np.power(centroids - data_i, 2), 1)
    return dist # dist^2

# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    m = dataSet.shape[0]
    choice_idx = np.random.choice(m, k, replace=False)
    centroids = dataSet[choice_idx]

    return centroids


def kmeans_plus_plus_Center(dataset, k):
    # centroids=[]
    total=0
    #首先随机选一个中心点
    firstCenter=np.random.choice(range(dataset.shape[0]))
    centroids = dataset[firstCenter,:]
    centroids = centroids.reshape(1,-1)
    #选择其它中心点，对于每个点找出离它最近的那个中心点的距离
    for i in range(0, k-1):
        weights=[np.min(distEclud(centroids, dataset[i])) for i in range(dataset.shape[0])]
        total=sum(weights)
        #归一化0到1之间
        weights=[x/total for x in weights]

        num=np.random.random()
        total=0
        x=-1
        while total<num:
            x+=1
            total+=weights[x]
        a = dataset[x, :].reshape(1, -1)
        centroids = np.append(centroids, dataset[x,:].reshape(1,-1), axis=0)
    # self.centroids=[[self.data[i][r] for i in range(1,self.cols)] for r in centroids]
    return np.array(centroids)


# k-means 聚类算法
def kMeans(dataSet, k, max_iter, dist_2_threshold, init, random_state):
    np.random.seed(random_state)
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m, 2))
    # if init == 'random':
    #     centroids = randCent(dataSet, k)
    # elif init== 'kmeans++':
    #     centroids = kmeans_plus_plus_Center(dataSet, k)
    # else:
    #     pass
    centroids = init
    print(centroids)
    centroids_hids = [[] for _ in range(k)]
    centroids_cp = deepcopy(centroids)
    for cent in range(k):
        centroids_hids[cent].append(centroids_cp[cent, :])

    clusterChanged = True
    num_iter = 0
    while clusterChanged:

        for i in range(m):

            dist = distEclud(centroids, dataSet[i, :])
            minIndex = np.argmin(dist)
            minDist = dist[minIndex]
            clusterAssment[i, :] = minIndex, minDist**2

        clusterChanged = False
        for cent in range(k):

            ptsInClust = dataSet[clusterAssment[:, 0] == cent]
            new_centroid = np.mean(ptsInClust, axis=0)
            centroids_hids[cent].append(new_centroid)

            new_centroid = new_centroid.reshape(1,-1)
            # dist^2
            dist_2 = distEclud(new_centroid, centroids[cent, :])

            if dist_2 > dist_2_threshold:
                centroids[cent, :] = new_centroid
                clusterChanged = True
            else:
                pass
                # print(dist_2)
        num_iter +=1
        if num_iter >= max_iter:
            print("iteration > 100")
            break
    # print(num_iter)
    return centroids, clusterAssment, np.array(centroids_hids), num_iter


# --------------------测试----------------------------------------------------
# 用测试数据及测试kmeans算法
# X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=random_state)
# start_t = time()
# centroids, clusterAssment = KMeans(X, n_centers)
# print("t: {}s".format(time()-start_t))
# SSE = np.sum(clusterAssment[:,1])
# print(SSE)
#
# plt.subplot(122)
# plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0])
# plt.title("Result of K-Means")
# plt.show()