from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class ColumnConcatenator(BaseEstimator, TransformerMixin):
    """
    This is a simple SK learn compatible feature transformer
    which will transform concatenate two columns into one.
    The resulting column will be categorical as the contents
    will be coerced into strings
    """
    def __init__(self, input_column_names):
        self.input_column_names = input_column_names
        if len(self.input_column_names)!=2:
           print("ERROR")
           exit(1)
        self.column_one = self.input_column_names[0]
        self.column_two = self.input_column_names[1]
        self.output_name = self.column_one + "_" + self.column_two
        self.output_names = [self.output_name]

    def fit( self, X, y = None ):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(data=X, columns = self.input_column_names)
        def merge_names(row):
            return row[self.column_one]+ "-" + row[self.column_two]
        df[self.output_name] = df.apply(merge_names, axis=1)
        return df.loc[:,self.output_names]
