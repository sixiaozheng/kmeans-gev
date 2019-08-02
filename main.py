import numpy as np
import matplotlib.pyplot as plt
from time import time, strftime
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import datasets
import kmeans_py
import kmeans_gev
from sklearn.preprocessing import StandardScaler
import random

from copy import deepcopy


random_state = 552#random.randint(1,1000)
print(random_state)
n_features = 2
n_centers = 5
n_samples = 900 * n_centers
block_size = 150 # sqrt(900) [5, 30, 150]
n_blocks = int(n_samples / block_size)
dist_threshold = 0.1
max_iter = 200
POT_top = 0.1
# dist_2_threshold = 0.00001

centers = [(-5, -5), (0, 0), (5, 5)]

np.random.seed(random_state)

# plt.ion()
plt.figure(figsize=(32, 18))
plt.suptitle(
    r"sample:{} centers:{} feature:{} block size:{} distance $\xi$:{} max iter:{} POT top:{} seed:{}".format(n_samples,
                                                                                                             n_centers,
                                                                                                             n_features,
                                                                                                             block_size,
                                                                                                             dist_threshold,
                                                                                                             max_iter,
                                                                                                             POT_top,
                                                                                                             random_state))
# datatset
# centers = [1.5, 1.5], [-1.5, -1.5]
# X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=[0.5, 1.5])

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=random_state)

# centers = [[1, 1], [-1, -1], [1, -1]]
# X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.6)

# t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
# x = t * np.cos(t)
# y = t * np.sin(t)
#
# X = np.concatenate((x, y))
# X += .7 * np.random.randn(2, n_samples)
# X = X.T

# n_samples = 2000
# X, y = datasets.make_circles(n_samples=n_samples, factor=.5,
#                                       noise=.05)

# X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
# X,y = datasets.make_blobs(n_samples=n_samples, random_state=8)
# X,y = np.random.rand(n_samples, 2), None
#
# # Anisotropicly distributed data
# random_state = 170
# X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
# transformation = [[0.6, -0.6], [-0.4, 0.8]]
# X_aniso = np.dot(X, transformation)
# X, y = X_aniso, y
#
# # blobs with varied variances
# X,y = datasets.make_blobs(n_samples=n_samples,
#                              cluster_std=[1.0, 2.5, 0.5],
#                              random_state=random_state)

X = StandardScaler().fit_transform(X)


def distEclud(centroids, data_i):
    # return np.sqrt(np.sum(np.power(vecA - vecB, 2)))  # 求两个向量之间的距离
    # dist = np.sqrt(np.sum(np.power(centroids - data_i, 2),1))
    dist = np.sqrt(np.sum(np.power(centroids - data_i, 2), 1))
    return dist # dist^2


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



centroids_random = randCent(X, n_centers)

centroids_kmeans = kmeans_plus_plus_Center(X, n_centers)



# sklearn kmeans
init_random_sk = deepcopy(centroids_random)
start_t = time()
kmeans = KMeans(n_clusters=n_centers, init=init_random_sk, n_init=1, precompute_distances=False, verbose=10, algorithm='full', random_state=random_state).fit(X)
t = time() - start_t

plt.subplot(241)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=169, linewidths=6, color='k',
            zorder=10)
plt.title("sklearn K-Means random t:{:.2f}s, MSE:{:.2f} n_iter:{}".format(t, kmeans.inertia_ / X.shape[0], kmeans.n_iter_))


# sklearn kmeans + kmeans++
init_kmeans_sk = deepcopy(centroids_kmeans)
start_t = time()
kmeans = KMeans(n_clusters=n_centers, init=init_kmeans_sk, n_init=1, precompute_distances=False, verbose=10, algorithm='full', random_state=random_state).fit(X)
t = time() - start_t

plt.subplot(245)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=169, linewidths=6, color='k',
            zorder=10)
plt.title("sklearn K-Means kmeans++ t:{:.2f}s, MSE:{:.2f} n_iter:{}".format(t, kmeans.inertia_ / X.shape[0], kmeans.n_iter_))


# kmeans python random
init_random_py = deepcopy(centroids_random)
start_t = time()
init_cent = 'random'
centroids, clusterAssment, centroids_hids, num_iter = kmeans_py.kMeans(X, n_centers, max_iter, dist_threshold,
                                                                       init_random_py, random_state)
t = time() - start_t
# print("t: {}s".format(time()-start_t))
MSE = np.sum(clusterAssment[:, 1]) / X.shape[0]

plt.subplot(242)
plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=169, linewidths=6, color='k', zorder=10)

for centroids in centroids_hids:
    plt.plot(centroids[:, 0], centroids[:, 1], 'ko-', linewidth=2)
    # print(centroids[:, 0], centroids[:, 1])

plt.title("py K-Means random t:{:.2f}s, MSE:{:.2f}, n_iter:{}".format(t, MSE, num_iter))


# kmeans python kmeans++
init_kmeans_py = deepcopy(centroids_kmeans)
start_t = time()
init_cent = 'kmeans++'
centroids, clusterAssment, centroids_hids, num_iter = kmeans_py.kMeans(X, n_centers, max_iter, dist_threshold,
                                                                       init_kmeans_py, random_state)
t = time() - start_t
# print("t: {}s".format(time()-start_t))
MSE = np.sum(clusterAssment[:, 1]) / X.shape[0]

plt.subplot(246)
plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=169, linewidths=6, color='k', zorder=10)

for centroids in centroids_hids:
    plt.plot(centroids[:, 0], centroids[:, 1], 'ko-', linewidth=2)
    # print(centroids[:, 0], centroids[:, 1])

plt.title("py K-Means kmeans++ t:{:.2f}s, MSE:{:.2f}, n_iter:{}".format(t, MSE, num_iter))

# Kmeans+GEV+BMM+random
extreme_model = 'BMM'
init_cent = 'random'
init_random_gev = deepcopy(centroids_random)
start_t = time()
centroids, clusterAssment, centroids_hids, num_iter, mean_fit_time, p0_samples, p1_samples = kmeans_gev.kMeans(X,
                                                                                                               n_centers,
                                                                                                               n_blocks,
                                                                                                               max_iter,
                                                                                                               dist_threshold,
                                                                                                               extreme_model,
                                                                                                               init_random_gev,
                                                                                                               POT_top,
                                                                                                               random_state)
end_t = time()

# evaluation
MSE_gev = np.sum(clusterAssment[:, -1]) / X.shape[0]

plt.subplot(243)
# clustering result
plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
# centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=169, linewidths=6, color='k', zorder=10)
# centroids path
for centroids in centroids_hids:
    plt.plot(centroids[:, 0], centroids[:, 1], 'ko-', linewidth=2)
    # print(centroids[:, 0], centroids[:, 1])
# P=0 or p=1 samples
# if p0_samples.shape[0]:
#     plt.scatter(p0_samples[:, 0], p0_samples[:, 1], c='k', marker='^', s=8)
# if p1_samples.shape[0]:
#     plt.scatter(p1_samples[:, 0], p1_samples[:, 1], c='k', marker='s', s=8)


plt.title(
    "K-Means+GEV random t:{:.2f}s, MSE:{:.2f}, n_iter:{}, m_fit_t:{:.2f}s".format(end_t - start_t, MSE_gev, num_iter,
                                                                                  mean_fit_time))

# Kmeans+GEV+BMM+kmeans++
extreme_model = 'BMM'
init_cent = 'kmeans++'
init_kmeans_gev = deepcopy(centroids_kmeans)
start_t = time()
centroids, clusterAssment, centroids_hids, num_iter, mean_fit_time, p0_samples, p1_samples = kmeans_gev.kMeans(X,
                                                                                                               n_centers,
                                                                                                               n_blocks,
                                                                                                               max_iter,
                                                                                                               dist_threshold,
                                                                                                               extreme_model,
                                                                                                               init_kmeans_gev,
                                                                                                               POT_top,
                                                                                                               random_state)
end_t = time()

# evaluation
MSE_gev = np.sum(clusterAssment[:, -1]) / X.shape[0]

plt.subplot(247)
# clustering result
plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
# centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=169, linewidths=6, color='k', zorder=10)
# centroids path
for centroids in centroids_hids:
    plt.plot(centroids[:, 0], centroids[:, 1], 'ko-', linewidth=2)
    # print(centroids[:, 0], centroids[:, 1])
# P=0 or p=1 samples
# if p0_samples.shape[0]:
#     plt.scatter(p0_samples[:, 0], p0_samples[:, 1], c='k', marker='^', s=8)
# if p1_samples.shape[0]:
#     plt.scatter(p1_samples[:, 0], p1_samples[:, 1], c='k', marker='s', s=8)

plt.title(
    "K-Means+GEV kmeans++ t:{:.2f}s, MSE:{:.2f}, n_iter:{}, m_fit_t:{:.2f}s".format(end_t - start_t, MSE_gev, num_iter,
                                                                                    mean_fit_time))

# Kmeans+GPD+POT random
extreme_model = 'POT'
init_cent = 'random'
init_random_gpd = deepcopy(centroids_random)
start_t = time()
centroids, clusterAssment, centroids_hids, num_iter, mean_fit_time, p0_samples, p1_samples = kmeans_gev.kMeans(X,
                                                                                                               n_centers,
                                                                                                               n_blocks,
                                                                                                               max_iter,
                                                                                                               dist_threshold,
                                                                                                               extreme_model,
                                                                                                               init_random_gpd,
                                                                                                               POT_top,
                                                                                                               random_state)
end_t = time()

# evaluation
MSE_gev = np.sum(clusterAssment[:, -1]) / X.shape[0]

plt.subplot(244)
# clustering result
plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
# centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=169, linewidths=6, color='k', zorder=10)
# centroids path
for centroids in centroids_hids:
    plt.plot(centroids[:, 0], centroids[:, 1], 'ko-', linewidth=2)
    # print(centroids[:, 0], centroids[:, 1])
# P=0 or p=1 samples
if p0_samples.shape[0]:
    plt.scatter(p0_samples[:, 0], p0_samples[:, 1], c='k', marker='^', s=8)
if p1_samples.shape[0]:
    plt.scatter(p1_samples[:, 0], p1_samples[:, 1], c='k', marker='s', s=8)

plt.title(
    "K-Means+GPD random t:{:.2f}s, MSE:{:.2f}, n_iter:{}, m_fit_t:{:.2f}s".format(end_t - start_t, MSE_gev, num_iter,
                                                                                  mean_fit_time))

extreme_model = 'POT'
init_cent = 'kmeans++'
init_kmeans_gpd = deepcopy(centroids_kmeans)
start_t = time()
centroids, clusterAssment, centroids_hids, num_iter, mean_fit_time, p0_samples, p1_samples = kmeans_gev.kMeans(X,
                                                                                                               n_centers,
                                                                                                               n_blocks,
                                                                                                               max_iter,
                                                                                                               dist_threshold,
                                                                                                               extreme_model,
                                                                                                               init_kmeans_gpd,
                                                                                                               POT_top,
                                                                                                               random_state)
end_t = time()

# evaluation
MSE_gev = np.sum(clusterAssment[:, -1]) / X.shape[0]

plt.subplot(248)
# clustering result
plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
# centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=169, linewidths=6, color='k', zorder=10)
# centroids path
for centroids in centroids_hids:
    plt.plot(centroids[:, 0], centroids[:, 1], 'ko-', linewidth=2)
    # print(centroids[:, 0], centroids[:, 1])
# P=0 or p=1 samples
if p0_samples.shape[0]:
    plt.scatter(p0_samples[:, 0], p0_samples[:, 1], c='k', marker='^', s=8)
if p1_samples.shape[0]:
    plt.scatter(p1_samples[:, 0], p1_samples[:, 1], c='k', marker='s', s=8)

plt.title(
    "K-Means+GPD kmeans++ t:{:.2f}s, MSE:{:.2f}, n_iter:{}, m_fit_t:{:.2f}s".format(end_t - start_t, MSE_gev, num_iter,
                                                                                    mean_fit_time))
timestr = strftime("%m%d%H%M", time.localtime())
plt.savefig("results/{}_r{}_s{}_c{}_b{}.jpg".format(timestr, random_state, n_samples, n_centers, block_size))
plt.show()
print(random_state)
