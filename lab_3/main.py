#from collections import Counter

import numpy as np
import filecmp
import pandas as pd

# запись классов в файл
def write_to_file(predictions, filename):
    with open(filename, 'w') as file:
        result = ''
        
        for i in range(len(predictions)):
            result += str(predictions[i]) + '\n'

        file.write(result)

class KNN:
    def __init__(self, k = 5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def prediction(self, X):
        y_pred = []

        for x in X:

            distances = []

            # вычисление евклидового расстояния между x и x_train
            for x_train in self.X_train:

                distances.append(np.sqrt(np.sum((x - x_train) ** 2)))
            
            # получаем индексы элементов отсортированного массива в порядке возрастания
            k_idx = np.argsort(distances)
            # выбираем только первые k элементов, чтобы учитывать только минимальные расстояния
            k_idx = k_idx[: self.k]
            
            # извлекаем метку из массива y_train и присваиваем их в k_neighbor_labels. Получили метки классов для knn
            k_neighbor_labels = [self.y_train[i] for i in k_idx]
            
            k_neighbor_labels_count = {} # словарь, в котором каждому уникальному значению соседей соответствует количество его повторений
            for label in k_neighbor_labels: # проверка на наличие меток в словаре
                if label in k_neighbor_labels_count: # если значение есть в словаре
                    k_neighbor_labels_count[label] += 1
                else: # если еще нет
                    k_neighbor_labels_count[label] = 1

            most_common_label = None # наиболее частая метка
            max_count = 0
            for label, count in k_neighbor_labels_count.items():
                if count > max_count: # если количество повторений больше максимального количества
                    max_count = count # обновляем максимальное количество
                    most_common_label = label # обновляем значение с наибольшим количеством повторений

            y_pred.append(most_common_label)

        return np.array(y_pred)

if __name__ == "__main__":

    file_path = 'lab_3\input.txt'
    data = pd.read_table(file_path, sep=',')
    X_test = data[data.columns[0:4]].loc[(data['TT'] == 'test')].values
    X_train = data[data.columns[0:4]].loc[(data['TT'] == 'train')].values
    y_train = data[data.columns[4]].loc[(data['TT'] == 'train')].values

    obj = KNN()

    obj.fit(X_train, y_train)

    predictions = obj.prediction(X_test)

    write_to_file(predictions, 'lab_3\output.txt')

    print(filecmp.cmp('lab_3\output.txt','lab_3\sample.txt'))
