import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

boston_data = fetch_openml(name="boston", as_frame=True, parser="pandas", version=1)
boston = boston_data.frame

names = ["INDUS", "DIS", "NOX", "MEDV", "LSTAT", "AGE", "RAD"]
data_train = boston[names]
scaler = StandardScaler()
data_train_matrix = scaler.fit_transform(data_train)

som = MiniSom(
    9,
    6,
    data_train_matrix.shape[1],
    sigma=0.7,
    learning_rate=0.2,
    neighborhood_function="gaussian",
    random_seed=123,
)

errors = []
for _ in range(500):
    som.train_random(data_train_matrix, 1)
    errors.append(som.quantization_error(data_train_matrix))

plt.figure(figsize=(10, 5))
plt.plot(errors, label="Изменения ошибки")
plt.xlabel("Итерация")
plt.ylabel("Ошибка квантования")
plt.legend()
plt.show()


def cool_blue_hot_red(n):
    return plt.cm.coolwarm(np.linspace(0, 1, n))


plt.figure(figsize=(10, 5))
plt.pcolor(som.distance_map(), cmap="coolwarm")
plt.colorbar(label="Среднее расстояние до прототипов")
plt.title("Quality")
plt.show()

win_map = som.win_map(data_train_matrix)
node_assignments = np.array([som.winner(x) for x in data_train_matrix])
boston["NODE"] = [n[0] * 6 + n[1] for n in node_assignments]

print(pd.Series(boston["NODE"]).value_counts())

mydata = som.get_weights().reshape(-1, data_train_matrix.shape[1])
clusters = fcluster(linkage(mydata), 5, criterion="maxclust")

plt.figure(figsize=(10, 5))
plt.pcolor(som.distance_map(), cmap="coolwarm")
for x, y in node_assignments:
    node_index = x * 6 + y
    if node_index < len(clusters):
        plt.text(
            y + 0.5, x + 0.5, str(clusters[node_index]), color="black", ha="center", va="center"
        )
plt.colorbar(label="Cluster ID")
plt.title("Кластеры узлов SOM")
plt.show()

print(clusters)
print(data_train)
print(som)
