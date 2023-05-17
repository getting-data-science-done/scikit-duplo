Usage Guide
===========


Python Package Usage
^^^^^^^^^^^^^^^^^^^^

You can import the scikit-duplo package within python and then make use of the
SciKit Learn Compatible Transformer for your ML Pipeline.

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
and out of sample fashion, then uses the outputs of the quantile regressors as a set
of new features.
 
