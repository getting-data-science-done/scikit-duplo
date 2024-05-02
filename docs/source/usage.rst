Usage Guide
===========


Python Package Usage
^^^^^^^^^^^^^^^^^^^^

You can import the scikit-duplo package within python and then make use of the
SciKit Duplo components inside your ML Pipelines.

QuantileStackRegressor
**********************

The QuantileStackRegressor is a meta learner that performs a regression task by
learning interal quantiles over the target variable as a set of new features.
In the example below we include a QuantileStackRegressor in a prediction pipeline.

.. code-block:: python

    from skduplo.meta import QuantileStackRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import RandomForestClassifier

    qsr = QuantileStackRegressor(
        classifier=RandomForestClassifier(),
        regressor=ExtraTreesRegressor(),
        cuts = [0, 50, 100, 200]
    )

This model learns a set of internal classifiers that cut the training data by the 
regression target value. In a sense the model learns a quantile regression stack in
an out-of-sample fashion, then uses the outputs of the quantile regressors as a set
of new features.
 
MultiStackRegressor
**********************

The MultiStackRegressor is a meta learner that extends the QuantileStackRegressor
by learning a set of out-of-sample regressors as well as the quantile models as
internal features.
In the example below we include a MultiStackRegressor in a prediction pipeline.

.. code-block:: python

    from skduplo.meta import MultiStackRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestClassifier

    msr = MultiStackRegressor(
        classifier=RandomForestClassifier(),
        regressor_list=[ExtraTreesRegressor(), RandomForestRegressor()]
        regressor=ExtraTreesRegressor(),
        cuts = [0, 50, 100, 200]
    )

Note that the `regressor_list` parameter is an arbitrary list of sklearn compatible regressors
that the model will train internally on cross validated data to make intermediate predictions.
