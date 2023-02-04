"""
====================
2. MLP
====================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

from ai4water.utils.utils import get_version_info
from SeqMetrics import RegressionMetrics

from utils import evaluate_model, get_dataset, make_data

# %%

get_version_info()

# %%

dataset , *_ = get_dataset(encoding="ohe")
X_train, y_train = dataset.training_data()
X_test, y_test = dataset.test_data()
original_data, *_ = make_data()

# %%

print(original_data.columns[:-1])

# %%
# While there is one target, which is listed below

# %%

print(original_data.columns[-1])

# %%
# MLP from scratch
# ----------------

# %%

dataset , *_ = get_dataset(encoding="le")
X_train, y_train = dataset.training_data()
X_test, y_test = dataset.test_data()

original_data, *_ = make_data()

hidden_units = [8, 8]
learning_rate = 0.006440897421063212
batch_size = 48

train_dataset = tf.data.Dataset.from_tensor_slices\
                     ((X_train,y_train)).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices\
                     ((X_train,y_train)).batch(batch_size)

def run_experiment(model, loss, train_dataset, test_dataset):

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")

FEATURE_NAMES = dataset.input_features

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

# %%
# standard neural network
# -----------------------

def create_baseline_model():
    # inputs = create_model_inputs()
    # input_values = [value for _, value in sorted(inputs.items())]
    # features = keras.layers.concatenate(input_values)
    inputs = layers.Input(shape=(27,), dtype=tf.float32, name='model_Inputs')
    features = layers.BatchNormalization()(inputs)
    #features = inputs
    # Create hidden layers with deterministic weights using the Dense layer.
    for units in hidden_units:
        features = layers.Dense(units, activation="relu")(features)
    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

num_epochs = 100
mse_loss = keras.losses.MeanSquaredError()
baseline_model = create_baseline_model()
run_experiment(baseline_model, mse_loss, train_dataset, test_dataset)

sample = 157
examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(sample))[
    0
]

predicted = baseline_model(examples).numpy()
# for idx in range(sample):
#     print(f"Predicted: {round(float(predicted[idx][0]), 1)} - Actual: {targets[idx]}")

metrics = RegressionMetrics(targets.numpy(), predicted)
print(f'test r2 = {metrics.r2()}')
print(f'test r2_score = {metrics.r2_score()}')