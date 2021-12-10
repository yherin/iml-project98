from time import sleep
from typing import Type
import pandas as pd
from pandas import DataFrame

def writeFile( filePath: str, data: DataFrame, modelScore: float):
    """
    filePath: path to write to
    data: Dataframe containing 'class4' and 'p' columns
    modelScore: estimate binary classification accuracy
    """
    def _run_tests( data, modelScore):
        if not type(modelScore) is float:
            raise TypeError('pass float to modelScore')
        if not isinstance(data, DataFrame):
            raise TypeError('pass data as a pandas dataframe')
        if 'class4' not in data.columns:
            raise AttributeError('class4 column is missing')
        if 'p' not in data.columns:
            raise AttributeError('p column is missing')
        if len(data.columns) > 2:
            raise AttributeError('pass only two columns: class4 and p')
        if data['p'].gt(1.0).any():
            raise AttributeError('some probabilities in the dataframe are > 0.99')
        if data['p'].lt(0).any():
            raise AttributeError('some probabilities in the dataframe are negative')
        if not (data['class4'].isin(['Ia', 'Ib', 'II', 'nonevent']).all()):
            raise AttributeError('incorrect class labels found')
            #only labels 'Ia', 'Ib', 'II', 'nonevent' are valid
    _run_tests(data, modelScore)

    fo = open(filePath, 'w', encoding='UTF-8')
    fo.write(f'{modelScore}\n')
    fo.close()
    sleep(1)
    fo = open(filePath, 'a', encoding='UTF-8')
    data.to_csv(fo, sep=',', index=False, index_label=False)
    fo.close()
    print(f'File written to {filePath}')

#testing
filePath='test.csv'
dataz = pd.DataFrame({'class4': ['Ia', 'Ib', 'II', 'Ia', 'nonevent' ], 'p': [0.3, 0.4, 0.99, 0.01, 0.512125]})
writeFile(filePath, dataz, 0.3)