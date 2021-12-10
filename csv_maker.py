from time import sleep
from typing import Type
import pandas as pd
from pandas import DataFrame

class csv_writer:

    def writeFile(self, filePath: str, data: DataFrame, modelScore: float):
        self._run_tests(data, modelScore)

        fo = open(filePath, 'w', encoding='UTF-8')
        fo.write(f'{modelScore}\n')
        fo.close()
        sleep(1)
        fo = open(filePath, 'a', encoding='UTF-8')
        data.to_csv(fo, sep=',', index=False, index_label=False)
        fo.close()
        print(f'File written to {filePath}')

    def _run_tests(self, data, modelScore):
        if not type(modelScore) is float:
            raise TypeError('pass float to modelScore')
        if not isinstance(data, DataFrame):
            raise TypeError('pass data as a pandas dataframe')
        if 'class4' not in data.columns.names:
            raise AttributeError('class4 column is missing')
        if 'p' not in data.columns.names:
            raise AttributeError('p column is missing')
        if len(data.columns) > 2:
            raise AttributeError('pass only two columns: class4 and p')
        if data['p'].gt(1.0).any():
            raise AttributeError('some probabilities in the dataframe are > 0.99')
        if data['p'].lt(0).any():
            raise AttributeError('some probabilities in the dataframe are negative')
        if (data['class4'] not in ['Ia', 'Ib', 'II', 'nonevent']).any():
            raise AttributeError('incorrect class labels found')