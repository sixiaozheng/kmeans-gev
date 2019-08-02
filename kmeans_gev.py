import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from scipy.stats import genextreme
from scipy.stats import genpareto
from time import time
from copy import deepcopy


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


def BMM_data(data, n_blocks):
    dist_block = np.split(data, n_blocks)
    dist_block_max = np.max(dist_block,1)

    # dist_block_max = np.zeros(n_blocks)
    # for idx, block in enumerate(dist_block):
    #     dist_block_max[idx] = np.max(block)
    # print(dist_block_max_1==dist_block_max)
    # print(dist_block_max_1.shape, dist_block_max.shape)

    return dist_block_max


def POT(data, top):
    data.sort()
    data_len = data.shape[0]
    split_len = int(data_len*top)
    return data[-split_len:]


def kMeans(dataSet, k, n_blocks, max_iter, dist_threshold, extreme_model, init, POT_top, random_state):
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
        centroids_hids[cent].append(centroids_cp[cent,:])

    num_iter = 0
    fit_time = 0
    clusterChanged = True


    while clusterChanged:
        p0_samples = []
        p1_samples = []
        # genextreme param
        genextreme_param = np.zeros((k, 3))
        for cent in range(k):
            # dist_c = np.zeros(m)
            # for i in range(m):
            #     dist_c[i] = distMeans(centroids[cent, :], dataSet[i, :])
            dist_c = distEclud(dataSet, centroids[cent, :])
            # dist_c = np.sqrt(dist_c)
            if extreme_model == 'BMM':
                dist_block_max = BMM_data(dist_c, n_blocks)
            elif extreme_model == 'POT':
                dist_over_thre = POT(dist_c, POT_top)
            else:
                pass

            # fit genextreme
            if extreme_model == 'BMM':
                start_t = time()
                c, loc, scale = genextreme.fit(dist_block_max, loc=0)
                fit_time += time() - start_t
                genextreme_param[cent][0] = c
                genextreme_param[cent][1] = loc
                genextreme_param[cent][2] = scale
            elif extreme_model == 'POT':
                start_t = time()
                c, loc, scale = genpareto.fit(dist_over_thre)
                fit_time += time() - start_t
                genextreme_param[cent][0] = c
                genextreme_param[cent][1] = loc
                genextreme_param[cent][2] = scale
            else:
                pass

        for i in range(m):
            # start_t = time()
            # dist = distEclud(centroids, dataSet[i, :])
            # dist = np.sqrt(dist)
            # minIndex_1 = np.argmin(dist)
            # minDist_1 = dist[minIndex_1]
            #
            # prob_1 = genextreme.cdf(minDist_1, genextreme_param[minIndex_1][0], genextreme_param[minIndex_1][1],
            #                       genextreme_param[minIndex_1][2])
            # print("t: {}s".format(time()-start_t))

            start_t = time()
            minProb = np.inf
            minIndex = -1
            minDist = np.inf
            for cent in range(k):

                distJI = distEclud(centroids[cent, :].reshape(1,-1), dataSet[i, :])
                if extreme_model == 'BMM':
                    a= centroids[cent, :].reshape(1,-1)
                    b=dataSet[i, :]
                    prob = genextreme.cdf(distJI, genextreme_param[cent][0], genextreme_param[cent][1],
                                          genextreme_param[cent][2])
                    x_0 = genextreme.ppf(0,genextreme_param[cent][0], genextreme_param[cent][1],
                                          genextreme_param[cent][2])
                    x_1 = genextreme.ppf(0.999,genextreme_param[cent][0], genextreme_param[cent][1],
                                          genextreme_param[cent][2])
                    if prob == 0:
                        p0_samples.append(dataSet[i, :])
                        # print("i:{},k:{},d:{},x_0:{},x_1:{}".format(i, cent, distJI, x_0, x_1))
                        # print("prob==0")
                    elif prob == 1:
                        p1_samples.append((dataSet[i,:]))
                        # print("i:{},k:{},d:{},x_0:{},x_1:{}".format(i, cent, distJI, x_0, x_1))
                        # print("prob==1")
                elif extreme_model == 'POT':
                    a = centroids[cent, :].reshape(1, -1)
                    b = dataSet[i, :]
                    prob = genpareto.cdf(distJI, genextreme_param[cent][0], genextreme_param[cent][1],
                                          genextreme_param[cent][2])
                    x_0 = genpareto.ppf(0, genextreme_param[cent][0], genextreme_param[cent][1],
                                         genextreme_param[cent][2])
                    x_1 = genpareto.ppf(0.999, genextreme_param[cent][0], genextreme_param[cent][1],
                                         genextreme_param[cent][2])
                    if prob == 0:
                        p0_samples.append(dataSet[i, :])
                        # print("i:{},k:{},d:{},x_0:{},x_1:{}".format(i, cent, distJI, x_0, x_1))
                        # print("prob==0")
                    elif prob == 1:
                        p1_samples.append((dataSet[i,:]))
                        # print("i:{},k:{},d:{},x_0:{},x_1:{}".format(i, cent, distJI, x_0, x_1))
                        # print("prob==1")

                if prob < minProb:
                    minProb = prob
                    minIndex = cent
                    minDist = distJI
            # print("t: {}s".format(time() - start_t))


            # if clusterAssment[i, 0] != minIndex:
            #     clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2

        p0_filtered = []
        for idx, p0_sample in enumerate(p0_samples):
            p0_samples[idx] = p0_sample.tolist()

        for p0_sample in p0_samples:
            if p0_samples.count(p0_sample)>=2 and (p0_sample not in p0_filtered):
                p0_filtered.append(p0_sample)

        p0_filtered = np.array(p0_filtered)

        p1_filtered = []
        for idx, p1_sample in enumerate(p1_samples):
            p1_samples[idx] = p1_sample.tolist()

        for p1_sample in p1_samples:
            if p1_samples.count(p1_sample) >= 2 and (p1_sample not in p1_filtered):
                p1_filtered.append(p1_sample)

        p1_filtered = np.array(p1_filtered)

        clusterChanged = False
        for cent in range(k):
            ptsInClust = dataSet[clusterAssment[:, 0] == cent]
            new_centroid = np.mean(ptsInClust, axis=0)
            centroids_hids[cent].append(new_centroid)

            new_centroid = new_centroid.reshape(1, -1)
            dist = distEclud(new_centroid, centroids[cent, :])
            # centroids[cent, :] = new_centroid

            if dist > dist_threshold:
                centroids[cent, :] = new_centroid
                clusterChanged = True
            else:
                pass

            # dist = distEclud(new_centroid, centroids[cent,:])
            # if dist > 0.0001:
            #     centroids[cent,:] = new_centroid
            # else:
            #     clusterChanged = True
        num_iter += 1
        if num_iter >= max_iter:
            print("iteration > 100")
            break
    # print(num_iter)
    mean_fit_time = fit_time/(num_iter*k)
    return centroids, clusterAssment, np.array(centroids_hids), num_iter, mean_fit_time, p0_filtered, p1_filtered


def main():
    # for n_blocks in [200, 250, 500]:
    n_samples = 1000
    random_state = 1
    n_features = 2
    n_centers = 3
    block_size = 20
    n_blocks = int(n_samples/block_size)
    dist_threshold = 0.1
    max_iter = 100
    extreme_model = 'BMM'
    centers = [(-5, -5), (0, 0), (5, 5)]

    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.6)

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=random_state)

    # X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)

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
    #########################################################################################
    # Kmeans
    start_t = time()
    kmeans = KMeans(n_clusters=n_centers, init='random', random_state=random_state).fit(X)
    end_t = time()

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_,s=2)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], marker='x', s=169, linewidths=3, color='k', zorder=10)
    plt.title("K-Means t:{:.2f}s, inertia:{:.2f}".format(end_t - start_t, kmeans.inertia_))

    #########################################################################################
    # Kmeans+GEV
    start_t = time()
    centroids, clusterAssment, centroids_hids = kMeans(X, n_centers, n_blocks, max_iter, dist_threshold, extreme_model)
    end_t = time()
    print(centroids)
    # evaluation
    inertia_gev = np.sum(clusterAssment[:, -1])

    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0],s=2)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='k', zorder=10)

    for centroids in centroids_hids:
        plt.plot(centroids[:,0], centroids[:,1],'ko-',linewidth=2)
        print(centroids[:,0], centroids[:,1])

    plt.title("K-Means+GEV t:{:.2f}s, inertia:{:.2f}".format(end_t - start_t, inertia_gev))
    plt.savefig("results/blobs_r{}_s{}_c{}_b{}.jpg".format(random_state, n_samples, n_centers, n_blocks))
    plt.show()


if __name__ == "__main__":
    main()
