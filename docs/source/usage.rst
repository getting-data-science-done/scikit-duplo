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

LookupEncoder
**********************

You can incorporate LookupEncoder at the pre-processing stage of your ML model. 
It creates an instance of a given lookup table and serialises the lookup table into the ML model. 

.. code-block:: python
    from skduplo.preprocessing import LookupEncoder

    lookup_table = {'A1': 1, 'A2': 2, 'A3': 3,'A4': 4, 
                    'B1': 5, 'B2': 6,'B3': 7, 'B4': 8}
    default_value = 4.5
    lookup_encoder = LookupEncoder(column_name='target_column', lookup_table=lookup_table, default_value=default_value)

This `lookup_encoder` can now be serialised in a ML model. 
At the training time or during inference, the model will map the elements of `target column` as per the given lookup_table.
If the `target_column` elements are not in the given list (in this case: A1, A2, A3, A4, B1, B2, B3, B4), then the `default_value` will be mapped instead.