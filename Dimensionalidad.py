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

# Ajustar PCA (2 componentes)
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

# Mostrar varianza explicada
explained = pca.explained_variance_ratio_
print("Varianza explicada por componente:")
for i, val in enumerate(explained, start=1):
    print(f"PC{i}: {val:.4f} ({val*100:.2f}%)")
print(f"Varianza total explicada (2 componentes): {explained.sum():.4f} ({explained.sum()*100:.2f}%)\n")

# Mostrar primeras filas transformadas
df = pd.DataFrame(X_r, columns=['PC1', 'PC2'])
df['target'] = y
df['target_name'] = df['target'].map(lambda t: target_names[t])

print("\nPrimeras filas:")
print(df.head(10))

# Gráfico de dispersión
plt.figure(figsize=(8,6))
for target in np.unique(y):
    mask = (y == target)
    plt.scatter(X_r[mask, 0], X_r[mask, 1], label=target_names[target])

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA (2 componentes) - Dataset Iris')
plt.legend()
plt.show()
