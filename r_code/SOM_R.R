install.packages("kohonen")
data(Boston, package = "MASS")
VarName = c("indus", "dis", "nox",  "medv", "lstat", "age", "rad")
# отбор переменных для обучения SOM
data_train <- Boston[, VarName]
data_train_matrix <- as.matrix(scale(data_train))

library(kohonen)
set.seed(123)
som_grid <- somgrid(xdim = 9, ydim = 6, topo = "hexagonal")
som_model <- som(data_train_matrix, grid = som_grid, rlen = 100,
                 alpha = c(0.05,0.01), keep.data = TRUE)
plot(som_model, type = "changes")



# Зададим палитру цветов
coolBlueHotRed <- function(n, alpha = 1) {
  rainbow(n, end = 4/6, alpha = alpha)[n:1]
}
par(mfrow = c(2, 1))
# Сколько объектов связано с каждым узлом?
plot(som_model, type = "counts", palette.name = coolBlueHotRed)
# Каково среднее расстояние объектов узла до его прототипов?
plot(som_model, type = "quality", palette.name = coolBlueHotRed)

colB <- ifelse(Boston$black <= 100, "red", "gray70")
par(mfrow = c(2, 1))
plot(som_model, type = "mapping", col = colB, pch = 16)
plot(som_model, type = "codes")

print(colB)

# Получение номеров узлов для каждого наблюдения
node_assignments <- som_model$unit.classif
print(node_assignments)
# Добавление номера узла в исходный датасет
Boston$node <- node_assignments

# Теперь вы можете видеть, какая строка соответствует какому узлу
# Например, чтобы увидеть строки, отнесённые к узлу 1:
node_1_rows <- Boston[Boston$node == 1, ]

# Вы можете также посмотреть количество наблюдений в каждом узле:
table(Boston$node)


par(mfrow = c(2, 1))
plot(som_model, type = "property",
     property = som_model$codes[[1]][,1],
     main = "indus - доля домов, продаваемых в розницу",
     palette.name = coolBlueHotRed)
var_unscaled <- aggregate(as.numeric(data_train[, 3]),
                          by = list(som_model$unit.classif),
                          FUN = mean, simplify = TRUE)[, 2]
plot(som_model, type = "property", property = var_unscaled,
     main = "nox - содержание окислов азота",
     palette.name = coolBlueHotRed)


## Формируем матрицу "узлы  переменные"
mydata <- as.matrix(som_model$codes[[1]])
# Используем иерархическую кластеризацию с порогом при k=5
som_cluster <- cutree(hclust(dist(mydata)), 5)
# Определяем палитру цветов
pretty_palette <- c("#1f77b4", '#ff7f0e', '#2ca02c',
                    '#d62728', '#9467bd', '#8c564b', '#e377c2')
# Показываем разными цветами кластеры узлов и переменные
plot(som_model, type = "codes",
     bgcol = pretty_palette[som_cluster])
add.cluster.boundaries(som_model, som_cluster)

print(som_cluster)
print(data_train)
print(som_model)
print(som_grid)
print
