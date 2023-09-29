import numpy as np # для математических вычислений
import pandas as pd # для графиков

def read_file(feature_0, feature_1, feature_2, feature_3, target, TT):
    # чтение данных из файла по столбцам
    file_path = 'lab_1\input.txt'
    with open(file_path, 'r') as file:
        file.readline() # скипаем первую строку
        for i in file:
            first, second, third, fourth, fifth, sixth = i.rstrip().split(',')
            feature_0.append(first)
            feature_1.append(second)
            feature_2.append(third)
            feature_3.append(fourth)
            target.append(fifth)
            TT.append(sixth)

def main():
    feature_0 = []
    feature_1 = []
    feature_2 = []
    feature_3 = []
    target = []
    TT = []
    read_file(feature_0, feature_1, feature_2, feature_3, target, TT)

if __name__ == '__main__':
    main()