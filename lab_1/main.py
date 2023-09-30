import numpy as np # для математических вычислений
import pandas as pd # для графиков и парсинга файла

def sigmoid(p):
    if (p == 1):
        return
    degree = p / (1 - p)
    return 1 / (1 + np.exp(-1 * degree))

def gradient_descent(x, y, iters = 1000, learning_rate = 0.0001, end = 1e-6):
    pass

def main():
    file_path = 'lab_1\input.txt'
    data = pd.read_table(file_path, sep=',')
    print(data)
    #print(sigmoid(-1))

    

if __name__ == '__main__':
    main()