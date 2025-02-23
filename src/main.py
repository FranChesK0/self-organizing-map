from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

SIGMA = 0.5
LEARNING_RATE = 0.001
ITERATIONS = 1000
RANDOM_SEED = 123
CLUSTERS = 5


def create_som(data, x, y, sigma=SIGMA, learning_rate=LEARNING_RATE, random_seed=RANDOM_SEED):
    return MiniSom(
        x,
        y,
        data.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        neighborhood_function="gaussian",
        random_seed=random_seed,
    )


root_dir_path = Path(__file__).parent.parent
dataset_path = Path(root_dir_path, "data", "Boston.csv")
boston = pd.read_csv(dataset_path)

columns = ["indus", "dis", "nox", "medv", "lstat", "age", "rad"]
data_train = boston[columns].copy()

scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train)

np.random.seed(RANDOM_SEED)
som_x, som_y = 9, 6

som = create_som(data_train_scaled, som_x, som_y)
som.random_weights_init(data_train_scaled)
q_error = []
for i in range(ITERATIONS):
    rand_i = np.random.randint(len(data_train_scaled))
    som.update(data_train_scaled[rand_i], som.winner(data_train_scaled[rand_i]), i, ITERATIONS)
    q_error.append(som.quantization_error(data_train_scaled))

plt.figure(figsize=(7, 7))
plt.plot(np.arange(ITERATIONS), q_error, label="quantization error")
plt.xlabel("iteration")
plt.ylabel("error")
plt.legend()
plt.show()

som = create_som(data_train_scaled, som_x, som_y)
som.random_weights_init(data_train_scaled)
som.train_random(data_train_scaled, ITERATIONS)

plt.figure(figsize=(8, 6))
frequencies = np.zeros((som_x, som_y))
for x in data_train_scaled:
    w = som.winner(x)
    frequencies[w] += 1
plt.imshow(frequencies.T, cmap="coolwarm", origin="lower")
plt.colorbar(label="Number of Observations")
plt.title("SOM Node Occupancy (Counts)")
plt.show()

plt.figure(figsize=(8, 6))
q_errors = np.zeros((som_x, som_y))
for x in data_train_scaled:
    w = som.winner(x)
    q_errors[w] += np.linalg.norm(x - som.get_weights()[w])
plt.imshow(q_errors.T, cmap="coolwarm", origin="lower")
plt.colorbar(label="Average Distance to Prototype")
plt.title("SOM Node Quality")
plt.show()

colors_boston = np.where(boston["black"] <= 100, "red", "gray")
plt.figure(figsize=(8, 6))
for i, x in enumerate(data_train_scaled):
    w = som.winner(x)
    plt.scatter(w[0] + 0.5, w[1] + 0.5, c=colors_boston[i], edgecolors="k", marker="o", s=50)
plt.xlim([0, som_x])
plt.ylim([0, som_y])
plt.title("SOM Mapping (Colored by 'black' Feature)")
plt.grid()
plt.show()

fig, ax = plt.subplots(som_x, som_y, figsize=(9, 6), dpi=100)
fig.suptitle("Code plot")
for x in range(som_x):
    for y in range(som_y):
        weights = som.get_weights()[x][y]
        ax[x, y].pie(
            np.abs(weights),
            labels=columns,
            colors=["green", "lime", "blue", "gold", "brown", "pink", "gray"],
        )
        ax[x, y].set_xticks([])
        ax[x, y].set_yticks([])
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()

node_assignments = np.array([som.winner(x) for x in data_train_scaled])
boston["node"] = [f"{x}-{y}" for x, y in node_assignments]
node_counts = boston["node"].value_counts()
print(node_counts)
node_1_rows = boston[boston["node"] == "1-1"]
print(node_1_rows)

nox_values = np.array(
    [
        data_train_scaled[:, columns.index("nox")][
            np.all(node_assignments == (i, j), axis=1)
        ].mean()
        if np.any(np.all(node_assignments == (i, j), axis=1))
        else 0
        for i in range(som_x)
        for j in range(som_y)
    ]
)
plt.figure(figsize=(8, 6))
plt.imshow(nox_values.reshape((som_x, som_y)), cmap="coolwarm", aspect="auto")
plt.colorbar(label="Среднее содержание NOx")
plt.title("nox - содержание окислов азота")
plt.show()

weights = som.get_weights().reshape(-1, len(columns))
linkage_matrix = linkage(weights, method="complete")
clusters = fcluster(linkage_matrix, 5, criterion="maxclust")

plt.figure(figsize=(8, 6))
plt.imshow(clusters.reshape((som_x, som_y)), aspect="auto")
plt.colorbar(label="Кластеры")
plt.title("Кластеризация узлов SOM")
plt.show()

weights = som.get_weights().reshape(-1, data_train_scaled.shape[1])
kmeans = KMeans(n_clusters=CLUSTERS, random_state=RANDOM_SEED)
clusters = kmeans.fit_predict(weights)

cluster_map = clusters.reshape((som_x, som_y))
plt.figure(figsize=(8, 8))
plt.imshow(cluster_map, cmap="Blues", alpha=0.7)
plt.colorbar(label="Cluster")

for x in range(som_x):
    for y in range(som_y):
        plt.text(y, x, cluster_map[x, y], ha="center", va="center", color="black")
plt.title("SOM Clusters")
plt.show()
