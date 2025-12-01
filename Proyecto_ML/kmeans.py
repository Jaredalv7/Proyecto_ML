from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. Generar datos de ejemplo
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=42)

# 2. Crear el modelo K-Means
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X)

# 3. Resultados
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 4. Gráfica
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], s=300, marker="X")
plt.title("Agrupación con K-Means")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()
