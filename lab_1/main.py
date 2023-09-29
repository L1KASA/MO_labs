import numpy as np # для математических вычислений
import pandas as pd # для графиков и парсинга файла

def main():
    file_path = 'lab_1\input.txt'
    data = pd.read_table(file_path, sep=',')
    

if __name__ == '__main__':
    main()