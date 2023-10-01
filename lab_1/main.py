import numpy as np # для математических вычислений
import pandas as pd # для графиков и парсинга файла

# https://www.dmitrymakarov.ru/opt/logistic-regression-05/#27-obuchenie-modeli
def sigmoid(p):
    if (p >= 0):
        return 1 / (1 + np.exp(-p))
    else:
        return np.exp(p) / (1 + np.exp(p))

def gradient(x, y, y_pred, n):
  return np.dot(x.T, (y_pred - y)) / n

def predict(x, thetas):
  # найдем значение линейной функции
  z = np.dot(x, thetas)
  # проведем его через устойчивую сигмоиду
  probs = np.array([sigmoid(value) for value in z])
  # если вероятность больше или равна 0,5 - отнесем наблюдение к классу 1, 
  # в противном случае к классу 0
  # дополнительно выведем значение вероятности
  return np.where(probs >= 0.5, 1, 0), probs

# metod naisc spuska
# https://academy.yandex.ru/handbook/ml/article/optimizaciya-v-ml
# https://github.com/YuryStrelkov/OptimizationMethods/commit/474a7871da78596b34dec615e1b0d3f940d11422
# https://design-hero.ru/articles/191445/

def gradient_descent(train, iters = 1000, learning_rate = 0.0001):
        coef = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            for row in train:
                yhat = predict(row, coef)
                error = row[-1] - yhat
                coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
                for i in range(len(row)-1):
                    coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
        return coef

def logistic_loss():
    pass

def fit(fun, x, y, iter = 20000, learning_rate = 0.001):

    pass 

def logistic_regression(feature_0, feature_1, feature_2, feature_3, target):
    pass

def main():
    file_path = 'lab_1\input.txt'
    data = pd.read_table(file_path, sep=',')
    print(data.groupby('target').mean())
    #print(sigmoid(-1))

    

if __name__ == '__main__':
    main()