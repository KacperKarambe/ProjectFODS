import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier


trainData = pd.read_csv("fashion-mnist_train.csv")
trainData = trainData.sample(n=10000, random_state=42)

x_train = trainData.iloc[:, 1:]
y_train = trainData.iloc[:, 0]

print(x_train.describe())
print(y_train.value_counts())

x_traindiv = np.divide(x_train, 255)

x_mean = x_traindiv.mean(axis=0)
Xtrn_nm = x_traindiv - x_mean

pca = PCA(n_components=2)
fitted_2D = pca.fit_transform(Xtrn_nm)

sc = plt.scatter(fitted_2D.T[0], fitted_2D.T[1], c=y_train, cmap = plt.cm.coolwarm)
clb = plt.colorbar(sc)
clb.ax.set_title('Class', fontsize=15)
plt.xlabel("Dimension 1", fontsize=12)
plt.ylabel("Dimension 2", fontsize=12)
plt.title('PCA 2D', fontsize=15)
plt.savefig("PCA.png")


tsne = TSNE(n_components = 2)
transformedTSNE = tsne.fit_transform(Xtrn_nm)

fig2 = plt.figure()
ax = plt.gca()

sc = plt.scatter(transformedTSNE[:,0], transformedTSNE[:,1], c=y_train, cmap = plt.cm.coolwarm)
clb = plt.colorbar(sc)
clb.ax.set_title('TSNE', fontsize=15)
plt.xlabel("Dimension 1", fontsize=12)
plt.ylabel("Dimension 2", fontsize=12)
plt.title('TSNE 2D', fontsize=15)
plt.savefig("TSNE.png")
    
print('Clustering normal Data')
kmeans = KMeans(n_clusters = 10, n_init = 35)
label = kmeans.fit(Xtrn_nm)
pred = kmeans.labels_


print('Accuracy of clustering normal data:', rand_score(y_train, pred))
metrics = ["euclidean","manhattan", "cosine"]
for i, metric in enumerate(metrics):
    aggClustering = AgglomerativeClustering(n_clusters=10, linkage="average", metric=metric ,compute_distances=True)
    model = aggClustering.fit(Xtrn_nm)
    print(metric + " rand index:", rand_score(y_train,aggClustering.labels_))

print('Clustering reduced Data')
reduced = pca.fit_transform(Xtrn_nm)
kmeans2 = KMeans(n_clusters = 10, n_init = 35)
kmeans2.fit(reduced)
pred_reduced = kmeans2.labels_

print('Accuracy of clustering reduced data:', rand_score(y_train, pred_reduced))
for i, metric in enumerate(metrics):
    aggClustering = AgglomerativeClustering(n_clusters=10, linkage="average", metric=metric ,compute_distances=True)
    model = aggClustering.fit(reduced)
    print(metric + " rand index:", rand_score(y_train,aggClustering.labels_))

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

x_traindiv = np.divide(x_train, 255)
x_testndiv = np.divide(x_test, 255)
x_mean_train = x_traindiv.mean(axis=0)
x_mean_test = x_testndiv.mean(axis=0)
Xtrn_nm = x_traindiv - x_mean_train
Xtst_nm = x_testndiv - x_mean_test

fold = KFold(n_splits=5)
d = {}
metrics = ["euclidean","manhattan","cosine"]
for metric in metrics:
    for n in range (1,10):
        accuracy = 0
        knn = KNeighborsClassifier(n_neighbors=n, metric=metric)
        knn.fit(Xtrn_nm, y_train)
        accuracy += knn.score(Xtst_nm, y_test)
        if metric in d.keys():
            d[metric].append(accuracy)
        else:
            d[metric] = [accuracy]
knn_results = pd.DataFrame(data=d)

mean = knn_results.mean()
print(knn_results)
print(knn_results.mean())
print(mean.mean())

tree = DecisionTreeClassifier(criterion='gini',random_state=42)
tree.fit(Xtrn_nm, y_train)
print('Classification: ', tree.score(Xtst_nm, y_test))
