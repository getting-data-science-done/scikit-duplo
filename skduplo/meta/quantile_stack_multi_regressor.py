import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor, is_classifier
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

class StackedMultiRegressor(BaseEstimator, RegressorMixin):
    """
    A meta regressor for doing model stacking for regression using underlying 
    quantile propensity models and internal regressors.
    Particularly designed for zero-inflated or heavily skewed datasets,

   `StackedMultiRegressor` consists of a series of classifiers, internal regressors
     all of which are fitted in an internal cross validation and scored out-of-sample
     A final regressor is trained over the original features and these stacked models.

    - The classifier's task is to build a series of propensity models that predict 
       if the target is above a given threshold. These are built in a two fold CV, 
       so that out of sample predictions
       can be added to the x vector for the final regression model
    - The regressor's task is to output the final prediction, aided by the 
       probabilities added by the underlying classifiers and internal regressors

    At prediction time, the average of the two classifiers is used for all 
      propensity models. Similarly the internal regressors are averaged.

    Parameters
    ----------
    classifier : Any, scikit-learn classifier

    regressor : Any, scikit-learn regressor
        A regressor for predicting the target. Its prediction is only used if `classifier` says that the output is non-zero.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    >>> np.random.seed(0)
    >>> X = np.random.randn(10000, 4)
    >>> y = ((X[:, 0]>0) & (X[:, 1]>0)) * np.abs(X[:, 2] * X[:, 3]**2)
    >>> z = StackedMultiRegressor(
    ... classifier=ExtraTreesClassifier(random_state=0),
    ... regressor=ExtraTreesRegressor(random_state=0)
    ... )
    >>> z.fit(X, y)
    StackedMultiRegressor(classifier=ExtraTreesClassifier(random_state=0),
                          regressor=ExtraTreesRegressor(random_state=0))
    >>> z.predict(X)[:5]
    array([4.91483294, 0.        , 0.        , 0.04941909, 0.        ])
    """

    def __init__(self, classifier, regressor_list, regressor, cuts=[0]) -> None:
        """Initialize."""
        self.classifier = classifier
        self.regressor_list = regressor_list
        self.regressor = regressor
        self.cuts = cuts

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.

        y : np.ndarray, 1-dimensional
            The target values.

        sample_weight : Optional[np.array], default=None
            Individual weights for each sample.

        Returns
        -------
        StackedRegressor
            Fitted regressor.

        Raises
        ------
        ValueError
            If `classifier` is not a classifier or `regressor` is not a regressor.
        """
        X, y = check_X_y(X, y)
        self._check_n_features(X, reset=True)

        if not is_classifier(self.classifier):
            raise ValueError(
                f"`classifier` has to be a classifier. Received instance of {type(self.classifier)} instead.")
        if not is_regressor(self.regressor):
            raise ValueError(f"`regressor` has to be a regressor. Received instance of {type(self.regressor)} instead.")

        """
          Now we need to internally split the data and build two sets of the classifiers to prevent leakage
        """
        X_ = [0] * 2
        y_ = [0] * 2
        X_[0], X_[1], y_[0], y_[1] = train_test_split(X, y, test_size=0.5)

        """
          Build two sets of classifiers for each of the required cuts
        """
        self.classifiers_ = [0] * 2

        for index in [0,1]:
            self.classifiers_[index] = [0] * len(self.cuts)
            for c, cut in enumerate(self.cuts):
                self.classifiers_[index][c] = clone(self.classifier)
                self.classifiers_[index][c].fit(X_[index], y_[index] > cut )

        """
          Build the internal regressors as well
        """
        self.regressors_ = [0] * 2

        for index in [0,1]:
            self.regressors_[index] = [0] * len(self.regressor_list)
            for c, reg in enumerate(self.regressor_list):
                self.regressors_[index][c] = clone(reg)
                self.regressors_[index][c].fit(X_[index], y_[index] )

        """
          Apply those classifier to the out of sample data
        """
        Xfinal_ = [0] * 2
        for index in [0,1]:
            Xfinal_[index] = X_[index].copy()
            c_index = 1 - index
            for c, cut in enumerate(self.cuts):
                preds = self.classifiers_[c_index][c].predict_proba( X_[index] )[:,1]
                Xfinal_[index] = np.append(Xfinal_[index], preds.T[:, None], axis=1)

        Xfinal_reg_ = [0] * 2
        for index in [0,1]:
            Xfinal_reg_[index] = Xfinal_[index].copy()
            c_index = 1 - index
            for c, reg in enumerate(self.regressor_list):
                preds = self.regressors_[c_index][c].predict( X_[index] )
                Xfinal_reg_[index] = np.append(Xfinal_reg_[index], preds.T[:, None], axis=1)

        """
          Join the split data into a final dataset for the regression model
        """
        Xfinale = np.concatenate((Xfinal_reg_[0], Xfinal_reg_[1] ), axis=0) 
        Yfinale = np.concatenate((y_[0], y_[1] ), axis=0)

        self.regressor_ = clone(self.regressor)
        self.regressor_.fit( Xfinale, Yfinale, sample_weight=sample_weight)
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

        """
          Apply classifiers to generate new colums
        """
        Xfinale = X.copy()

        for c, cut in enumerate(self.cuts):
            temp = np.zeros(len(X))
            for index in [0,1]:
                temp = temp + self.classifiers_[index][c].predict_proba(X)[:,1]
            temp = temp/2
            Xfinale = np.append(Xfinale, temp[:, None], axis=1)

        for c, reg in enumerate(self.regressor_list):
            temp = np.zeros(len(X))
            for index in [0,1]:
                temp = temp + self.regressors_[index][c].predict( X )
            temp = temp/2
            Xfinale = np.append(Xfinale, temp[:, None], axis=1)

        return self.regressor_.predict(Xfinale)
    

