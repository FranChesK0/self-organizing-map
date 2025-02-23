import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

# Загрузка данных
boston_data = fetch_openml(name="boston", as_frame=True, parser="pandas", version=1)
boston = boston_data.frame

# Выбор переменных
names = ["INDUS", "DIS", "NOX", "MEDV", "LSTAT", "AGE", "RAD"]
data_train = boston[names]
scaler = StandardScaler()
data_train_matrix = scaler.fit_transform(data_train)

# Создание SOM размером 9x6 узлов
som = MiniSom(
    9,
    6,
    data_train_matrix.shape[1],
    # sigma=0.7,
    # learning_rate=0.2,
    sigma=0.05,
    learning_rate=0.01,
    neighborhood_function="gaussian",
    random_seed=123,
    topology="hexagonal",
)

# Обучение SOM с сохранением ошибки
errors = []
for _ in range(100):
    som.train_random(data_train_matrix, 1)
    errors.append(som.quantization_error(data_train_matrix))

# Визуализация ошибки
plt.figure(figsize=(10, 5))
plt.plot(errors, label="Изменения ошибки")
plt.xlabel("Итерация")
plt.ylabel("Ошибка квантования")
plt.legend()
plt.show()


# Количество объектов на каждом узле
plt.figure(figsize=(10, 5))
plt.pcolor(som.distance_map(), cmap="coolwarm")
plt.colorbar(label="Среднее расстояние до прототипов")
plt.title("Quality")
plt.show()

# Присвоение узлов каждому наблюдению
win_map = som.win_map(data_train_matrix)
node_assignments = np.array([som.winner(x) for x in data_train_matrix])
boston["NODE"] = [n[0] * 6 + n[1] for n in node_assignments]

# Количество наблюдений в каджом узле
print(pd.Series(boston["NODE"]).value_counts())

# Визуализация кластеров
mydata = som.get_weights().reshape(-1, data_train_matrix.shape[1])
clusters = fcluster(linkage(mydata, method="complete"), 5, criterion="maxclust")

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
