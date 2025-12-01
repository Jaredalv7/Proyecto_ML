import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar datos
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# PCA a 2 componentes
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

# Varianza explicada
explained = pca.explained_variance_ratio_
print("Varianza explicada:")
print(f"PC1: {explained[0]:.4f}")
print(f"PC2: {explained[1]:.4f}")

# DataFrame
df = pd.DataFrame(X_r, columns=["PC1", "PC2"])
df["target"] = y
df["target_name"] = df["target"].map(lambda t: target_names[t])
print(df.head())

# Gráfica
plt.figure(figsize=(8,6))
for target in np.unique(y):
    plt.scatter(X_r[y == target, 0], X_r[y == target, 1], label=target_names[target])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Iris Dataset")
plt.legend()
plt.show()
