
# CLUSTERING METRICS PRACTICE


from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos (puedes cambiar por tus datos)
data = load_iris()
X = data.data

# Rango de clusters a evaluar
k_values = [2, 3, 4, 5, 6]

results = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)

    results.append([k, inertia, silhouette, dbi])

# Convertir a DataFrame
df_results = pd.DataFrame(results, columns=["Clusters", "Inercia", "Silhouette", "Davies-Bouldin"])

print(df_results)

# ----- Gráficas -----

plt.figure()
plt.plot(df_results["Clusters"], df_results["Silhouette"], marker="o")
plt.xlabel("Número de clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette vs Número de Clusters")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df_results["Clusters"], df_results["Inercia"], marker="o")
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.title("Inercia vs Número de Clusters")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df_results["Clusters"], df_results["Davies-Bouldin"], marker="o")
plt.xlabel("Número de clusters")
plt.ylabel("Davies-Bouldin Index")
plt.title("DBI vs Número de Clusters")
plt.grid(True)
plt.show()
