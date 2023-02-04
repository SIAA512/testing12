"""
====================
4. Utils
====================
"""
import os
from typing import Union, List, Tuple, Any

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import matplotlib.pyplot as plt
from ai4water.postprocessing import LossCurve

from SeqMetrics import RegressionMetrics

from ai4water.preprocessing import DataSet

def _ohe_column(df:pd.DataFrame, col_name:str)->tuple:
    # function for OHE
    assert isinstance(col_name, str)

    # setting sparse to True will return a scipy.sparse.csr.csr_matrix
    # not a numpy array
    encoder = OneHotEncoder(sparse=False)
    ohe_cat = encoder.fit_transform(df[col_name].values.reshape(-1, 1))
    cols_added = [f"{col_name}_{i}" for i in range(ohe_cat.shape[-1])]

    df[cols_added] = ohe_cat

    df.pop(col_name)

    return df, cols_added, encoder

def le_column(df:pd.DataFrame, col_name)->tuple:
    """label encode a column in dataframe"""
    encoder = LabelEncoder()
    df[col_name] = encoder.fit_transform(df[col_name])
    return df, encoder

def _load_data(input_features:list=None)->pd.DataFrame:

    # read excel
    # our data is on the first sheet of both files
    dirname = os.path.dirname(__file__)
    data = pd.read_excel(os.path.join(dirname, 'PO4 data.xlsx'))

    data = data.drop(columns=['Cf'])

    #removing original index of both dataframes and assigning a new index
    data = data.reset_index(drop=True)

    target = ['qe']

    if input_features is None:
        input_features = data.columns.tolist()[0:-1]
    else:
        assert isinstance(input_features, list)
        assert all([feature in data.columns for feature in input_features])

    return data[input_features + target]

def make_data(
        input_features:list = None,
        encoding:str = None
)->Tuple[pd.DataFrame, Any, Any, Any]:

    data = _load_data(input_features)

    adsorbent_encoder, fs_encoder, at_encoder  = None, None, None
    if encoding=="ohe":
        # applying One Hot Encoding
        data, _, adsorbent_encoder = _ohe_column(data, 'Adsorbent')
        data, _, fs_encoder = _ohe_column(data, 'Feedstock')
        data, _, at_encoder = _ohe_column(data, 'Anion_type')

    elif encoding == "le":
        # applying Label Encoding
        data, adsorbent_encoder = le_column(data, 'Adsorbent')
        data, fs_encoder = le_column(data, 'Feedstock')
        data, at_encoder = le_column(data, 'Anion_type')

    # moving target to last
    target = data.pop('qe')
    data['qe'] = target

    return data, adsorbent_encoder, fs_encoder, at_encoder

def get_dataset(encoding="ohe"):
    data, adsorbent_encoder, fs_encoder, at_encoder = make_data(encoding=encoding)

    dataset = DataSet(data=data,
                      seed=1575,
                      val_fraction=0.0,
                      split_random=True,
                      input_features=data.columns.tolist()[0:-1],
                      output_features=data.columns.tolist()[-1:],
                      )
    return dataset, adsorbent_encoder, fs_encoder, at_encoder


def box_violin(ax, data, palette=None):
    if palette is None:
        palette = sns.cubehelix_palette(start=.5, rot=-.5, dark=0.3, light=0.7)
    ax = sns.violinplot(orient='h', data=data,
                        palette=palette,
                        scale="width", inner=None,
                        ax=ax)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width, height / 2, transform=ax.transData))

    sns.boxplot(orient='h', data=data, saturation=1, showfliers=False,
                width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
    old_len_collections = len(ax.collections)

    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0, 0.12]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return

def evaluate_model(true, predicted):
    metrics = RegressionMetrics(true, predicted)
    for i in ['mse', 'rmse', 'r2', 'r2_score', 'mape', 'mae']:
        print(i, getattr(metrics, i)())
    return

def run_experiment(model, loss, train_dataset, test_dataset,
                   learning_rate, num_epochs):

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    h = model.fit(train_dataset, epochs=num_epochs,
                  validation_data=test_dataset, verbose=0)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")

    LossCurve().plot_loss(h.history)

    return

def create_model_inputs(FEATURE_NAMES):
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

# We use the tfp.layers.DenseVariational layer instead of
# the standard keras.layers.Dense layer in the neural
# network model.

def create_epistemic_bnn_model(train_size, hidden_units):
    # inputs = create_model_inputs()
    # features = keras.layers.concatenate(list(inputs.values()))
    inputs = layers.Input(shape=(27,), dtype=tf.float32, name='model_Inputs')
    features = layers.BatchNormalization()(inputs)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="relu",
        )(features)

    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def compute_predictions(model, examples,
                        iterations=100):
    predicted = []
    for _ in range(iterations):
        predicted.append(model.predict(examples, batch_size=len(examples),
                                       verbose=0))
    predicted = np.concatenate(predicted, axis=1)

    prediction_mean = np.mean(predicted, axis=1).tolist()
    prediction_min = np.min(predicted, axis=1).tolist()
    prediction_max = np.max(predicted, axis=1).tolist()
    prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist()

    df = pd.DataFrame()
    df['prediction'] = prediction_mean
    df['upper'] = prediction_max
    df['lower'] = prediction_min

    return df

def plot_ci(df, alpha, type):
    # plots the confidence interval

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill_between(np.arange(len(df)), df['upper'], df['lower'], alpha=0.5, color='C1')
    p1 = ax.plot(df['prediction'], color="C1", label="Prediction")
    p2 = ax.fill(np.NaN, np.NaN, color="C1", alpha=0.5)
    percent = int((1 - alpha) * 100)
    ax.legend([(p2[0], p1[0]), ], [f'{percent}% Confidence Interval'],
              fontsize=12)
    ax.set_xlabel(f"{type} Samples", fontsize=12)
    ax.set_ylabel("qe", fontsize=12)

    return ax

def create_aleatoric_bnn_model(train_size, hidden_units):
    # inputs = create_model_inputs()
    # features = keras.layers.concatenate(list(inputs.values()))
    inputs = layers.Input(shape=(27,), dtype=tf.float32, name='model_Inputs')
    features = layers.BatchNormalization()(inputs)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)

    # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.
    distribution_params = layers.Dense(units=2)(features)
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)