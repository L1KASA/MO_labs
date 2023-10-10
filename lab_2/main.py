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

class Naive_Bayes_Classifier:
    def __init__(self, n, mean_X, mean_Y):
        # инициализируем вес, смещение и вспомогательную переменную
        self.n = n
        #self.w = np.random.randn(n, 1) * 0.001
        #self.b = np.random.randn() * 0.001
        #self.report = 100
        # массивы для хранения результатов функции потерь
        #self.loss_history = []
        self.mean_Y = mean_Y
        self.mean_X = mean_X

    def finding_sigmoid(self, Xi):
        sigm = 0
        for i in range(len(Xi)):
            for j in range(len(Xi[0])):
                sigm += (Xi[i][j] - self.mean_Y)**2
        return (sigm / (self.n - 1)) ** 0.5
    

def main():
    file_path = 'lab_1\in1.txt'
    data = pd.read_table(file_path, sep=',')
    #test_set = data[data.columns[0:5]].loc[(data['TT'] == 'test')].values.tolist()
    #y_binar = data[data.columns[5]].loc[(data['TT'] == 'train')].values.tolist()
    #x_tr = data[data.columns[0:5]].loc[(data['TT'] == 'train')].values.tolist()
    x_tr1 = data[data.columns[0:5]].loc[(data['TT'] == 'train') & (data['target'] == 0)].values.tolist()
    #y_tr = data[data.columns[0:5]].loc[(data['TT'] == 'train') & (data['target'] == 0)].values.tolist()
    mu_X = data[data.columns[0:5]].loc[(data['TT'] == 'train') & (data['target'] == 1)].values.mean()
    mu_Y = data[data.columns[0:5]].loc[(data['TT'] == 'train') & (data['target'] == 0)].values.mean()
    #print(rr)
    #print(x_tr1)
    n = len(x_tr1[0]) # размер объекта
    #print(n)
    obj = Naive_Bayes_Classifier(n, mu_X, mu_Y)
    #obj.finding_sigmoid(x_tr1)
    #print(mu_X, mu_Y)
    print(obj.finding_sigmoid(x_tr1))

    #obj = Logistic_Regression(n) # создаем объект класса Logistic_Regression
    #obj.fit(x_tr, y_binar, iters=200)

    #predictions = obj.predict(test_set)

    #write_to_file(predictions, 'output.txt')

if __name__ == "__main__":
    main()