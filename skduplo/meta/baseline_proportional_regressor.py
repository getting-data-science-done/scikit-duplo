import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.exceptions import NotFittedError

class BaselineProportionalRegressor(BaseEstimator, RegressorMixin):
    """
    A meta regressor for learning the target value as a proportional difference
       relative to a mean value for a subset of other features.

    Creates and maintains an internal lookup table for the baseline during the
       model fir process.
 
    Parameters
    ----------
    regressor : Any, scikit-learn regressor that will be learned for the adjust target
    """

    def __init__(self, baseline_columns, regressor) -> None:
        """Initialize."""
        self.baseline_cols = baseline_columns
        self.regressor = regressor
        self.baseline_func = 'mean'

    def generate_baseline(self, df):
        self.lookup_table = df.groupby(self.baseline_cols).agg({'baseline':self.baseline_func}).reset_index()
        self.baseline_default = df['baseline'].agg(self.baseline_func)        

    def get_baseline_predictions(self, df):
        new_df = pd.merge(df, self.lookup_table, how='left', on=self.baseline_cols)
        new_df['baseline'] = np.where(new_df['baseline'].isnull(), self.baseline_default, new_df['baseline'])
        return new_df['baseline']

    def get_relative_target(self, baseline, y):
        return (y-baseline)/baseline
 
    def invert_relative_target(self, preds, baseline):
        return (preds*baseline)+baseline

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model.

        Note: this model diverges from the scikit-learn standard in that it needs a 
        pandas dataframe. 

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The training data.

        y : np.ndarray, 1-dimensional
            The target values.

        sample_weight : Optional[np.array], default=None
            Individual weights for each sample.

        Returns
        -------
        BaselineProportionalRegressor
            Fitted regressor.

        Raises
        ------
        ValueError
            If `regressor` is not a regressor.
        """
        X, y = check_X_y(X, y)
        self._check_n_features(X, reset=True)

        if not is_regressor(self.regressor):
            raise ValueError(f"`regressor` has to be a regressor. Received instance of {type(self.regressor)} instead.")

        for col in self.baseline_cols:
            if col not in X.columns
                raise ValueError(f"pandas.DataFrame required with baseline columns: `{col}` NOT FOUND.")

        df = X.copy()
        df['baseline'] = y
        self.generate_baseline(df)
        baseline = self.get_baseline_predictions(df)

        Yfinale = self.get_relative_target(baseline, y):
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit( X, Yfinale, sample_weight=sample_weight)
        return self


    def predict(self, X):
        """
        Get predictions.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples to get predictions of.

        Returns
        -------
        y : np.ndarray, shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        for col in self.baseline_cols:
            if col not in X.columns
                raise ValueError(f"pandas.DataFrame required with baseline columns: `{col}` NOT FOUND.")

        baseline = self.get_baseline_predictions(X)
        preds = self.regressor_.predict(Xfinale)

        return self.invert_relative_target(preds, baseline)
