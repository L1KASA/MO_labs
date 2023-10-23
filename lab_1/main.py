import numpy as np
import pandas as pd

# запись классов в файл
def write_to_file(predictions, filename):
    with open(filename, 'w') as file:
        result = ''
        
        # оценка предсказаний
        for i in range(len(predictions)):
            if predictions[i] > 0.5:
                result += '1\n'
            else:
                result += '0\n'

        file.write(result[:-1])

class Logistic_Regression:
    def __init__(self, n):
        # инициализируем вес, смещение и вспомогательную переменную
        self.n = n
        self.w = np.random.randn(n, 1) * 0.001
        self.b = np.random.randn() * 0.001
        self.report = 100
        # массивы для хранения результатов функции потерь
        self.loss_history = []

    def sigmoid(self, p): 
        if (p >= 0): # устойчива при очень больших положительных значениях p
            return 1 / (1 + np.exp(-p))
        else: # устойчива при очень больших отрицательных значениях p
            return np.exp(p) / (1 + np.exp(p))
        
    # Функция минимизации потерь, где y_true - значения из предоставленных данных, а y_pred - наше предсказание, значение которого пренадлежит промежутку от 0 до 1
    def loss_function(self, y_true, y_pred):
        for i in range(len(y_true)):
            y_one_loss = y_true[i] * np.log(y_pred[i] + 1e-9) # функция потерь для y = 1, добавив 1e-9, чтобы избежать ошибки при log(0)
            y_zero_loss = (1 - y_true[i]) * np.log(1 - y_pred[i] + 1e-9) # функция потерь для y = 0, , добавив 1e-9, чтобы избежать ошибки при log(0)
        return(-np.mean(y_zero_loss + y_one_loss)) # складываем и делим на количество наблюдений
    
    # Обучение модели
    def fit(self, X, Y, iters, regression_step = 0.005):

        # цикл, равный количеству операций
        for i in range(iters): 
            # инициализация производных
            d_w = np.zeros((self.n, 1))
            d_b = 0

            for j in range(len(X)):

                # вычисляем вес и сигмоиду
                z = np.dot(np.reshape(self.w, (1, self.n)), np.reshape(X[j], (self.n, 1))) + self.b
                a = self.sigmoid(z)

                # заполнение производных
                d_w += (a - Y[j]) * np.reshape(X[j], (self.n, 1))
                d_b += (a - Y[j])
            
            # нормируем производные и делаем градиентный шаг
            self.w -= np.dot(regression_step, d_w / self.n) 
            self.b -= np.dot(regression_step, d_b / self.n)

            # смотрим размер ошибки и добавляем в массив
            if i % self.report == 0:
                self.loss_history.append(self.loss_function(Y, self.predict(X)))
    
    # делаем прогноз с помощью обученной модели
    def predict(self, X):
        return np.array([self.sigmoid(np.dot(np.reshape(self.w, (1, self.n)), np.reshape(x, (self.n, 1))) + self.b) for x in X])

def main():
    file_path = 'lab_1\input.txt'
    data = pd.read_table(file_path, sep=',')
    test_set = data[data.columns[0:4]].loc[(data['TT'] == 'test')].values.tolist()
    y_binar = data[data.columns[4]].loc[(data['TT'] == 'train')].values.tolist()
    x_tr = data[data.columns[0:4]].loc[(data['TT'] == 'train')].values.tolist()

    n = len(x_tr[0]) # размер объекта
    obj = Logistic_Regression(n) # создаем объект класса Logistic_Regression
    obj.fit(x_tr, y_binar, iters=200)

    predictions = obj.predict(test_set)

    write_to_file(predictions, 'output.txt')

    

if __name__ == "__main__":
    main()