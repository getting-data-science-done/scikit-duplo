from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TwoColumnRatioDiff(BaseEstimator, TransformerMixin):
    """
    This is a simple SK learn compatible feature transformer
    which will transform two numerical columns into their ratio
    and difference
    """
    def __init__(self, input_column_names):
        self.input_column_names = input_column_names
        if len(self.input_column_names)!=2:
           print("ERROR")
           exit(1)
        self.column_one = self.input_column_names[0]
        self.column_two = self.input_column_names[1]
        self.diff_name = self.column_one + "-" + self.column_two
        self.ratio_name = self.column_one + "/" + self.column_two
        self.output_names = [self.ratio_name, self.diff_name]

    def fit( self, X, y = None ):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(data=X, columns = self.input_column_names)
        df[self.ratio_name] = np.where(
            df[self.column_two]>0,
            df[self.column_one]/df[self.column_two], 0.0
        )
        df[self.diff_name] = df[self.column_one]-df[self.column_two]
        return df.loc[:,self.output_names]

