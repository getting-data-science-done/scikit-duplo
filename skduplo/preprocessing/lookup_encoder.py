from sklearn.base import BaseEstimator, TransformerMixin

class LookupEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, lookup_table, default_value):
        self.column_name = column_name
        self.lookup_table = lookup_table
        self.default_value = default_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output_df = X.copy()
        output_df[self.column_name] = output_df[self.column_name].apply(lambda x: self.lookup_table.get(x, self.default_value))
        return output_df


