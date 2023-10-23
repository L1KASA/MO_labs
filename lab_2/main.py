import numpy as np
import pandas as pd
import filecmp
# запись классов в файл
def write_to_file(predictions, filename):
    with open(filename, 'w') as file:
        result = ''
        
        for i in range(len(predictions)):
            result += str(predictions[i]) + '\n'

        file.write(result[:-1])

class Naive_Bayes_Classifier:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y) # выявляем, какие классы имеются
        n_classes = len(self._classes)
        # init
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64) 
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for i in self._classes:
            X_c = X[y == i]
            self._mean[i, :] = X_c.mean(axis=0)
            self._var[i, :] = X_c.var(axis=0)
            self._priors[i] = X_c.shape[0] / float(n_samples)

    def prediction(self, X):     
        y_pred = [] # list с предсказаниями
        for x_item in X: # проходимся по признакам
            posteriors = []
        # вычисление апостериорной вероятности для каждого класса
            for i in self._classes:
                prior = np.log(self._priors[i])
                new_var = self.likelyhood(i, x_item)
                posterior = np.sum(np.log(new_var))
                posteriors.append(prior + posterior)
            # выбираем классовую переменную с максимальной вероятностью
            y_pred.append(self._classes[np.argmax(posteriors)]) 
        # возвращает класс с наибольшей апостериорной вероятностью
        return y_pred

    # функция правдоподобия возвращает вектор распределения Гаусса для конкретного X
    def likelyhood(self, class_i, x):
        mean = self._mean[class_i]
        var = self._var[class_i]
        return (1 / (np.sqrt(2 * np.pi * var))) * np.exp(-(x - mean)**2 / (2 * var))

def main():
    file_path = 'lab_2\input.txt'
    data = pd.read_table(file_path, sep=',')
    X_test = data[data.columns[0:4]].loc[(data['TT'] == 'test')].values
    X_train = data[data.columns[0:4]].loc[(data['TT'] == 'train')].values
    y_train = data[data.columns[4]].loc[(data['TT'] == 'train')].values

    obj = Naive_Bayes_Classifier()

    obj.fit(X_train, y_train)

    predictions = obj.prediction(X_test)

    write_to_file(predictions, 'lab_2\output1.txt')

    print(filecmp.cmp('lab_2\output.txt','lab_2\output1.txt'))

if __name__ == "__main__":
    main()