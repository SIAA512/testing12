"""
====================================================
3. Epistemic Bayesian neural network (epistemic_BNN)
====================================================
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from ai4water.utils.utils import get_version_info
from SeqMetrics import RegressionMetrics

from utils import  get_dataset, \
    make_data, run_experiment, \
    compute_predictions, create_epistemic_bnn_model, \
    plot_ci

# %%

get_version_info()

# %%

dataset , *_ = get_dataset(encoding="le")
X_train, y_train = dataset.training_data()
X_test, y_test = dataset.test_data()

# %%

original_data, *_ = make_data()

# %%

hidden_units = [8, 8]
learning_rate = 0.006440897421063212
batch_size = 48
num_epochs = 500
alpha = 0.05

# %%

train_dataset = tf.data.Dataset.from_tensor_slices\
                     ((X_train,y_train)).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices\
                     ((X_train,y_train)).batch(batch_size)

# %%

FEATURE_NAMES = dataset.input_features

# %%

bnn_model_full = create_epistemic_bnn_model(len(y_train)+len(y_test), hidden_units)

# %%

run_experiment(bnn_model_full, keras.losses.MeanSquaredError(), train_dataset,
               test_dataset, learning_rate, num_epochs)

# %%

tr_examples, tr_targets = list(train_dataset.unbatch().
                               shuffle(batch_size * 10).batch(len(y_train)))[0]

# %%

test_examples, test_targets = list(test_dataset.unbatch().
                                   shuffle(batch_size * 10).batch(len(y_test)))[0]

# %%

tr_predicted = compute_predictions(bnn_model_full, tr_examples)

# %%

tr_metrics = RegressionMetrics(tr_targets.numpy(),
                               tr_predicted['prediction'])
print(f'train r2 = {tr_metrics.r2()}')
print(f'train r2_score = {tr_metrics.r2_score()}')

# %%

plot_ci(tr_predicted, alpha, type='train')
plt.tight_layout()
plt.show()

# %%

plot_ci(tr_predicted.iloc[0:50], alpha, type='train')
plt.tight_layout()
plt.show()

# %%

test_predicted = compute_predictions(bnn_model_full, test_examples)

# %%

test_metrics = RegressionMetrics(test_targets.numpy(),
                                 test_predicted['prediction'])
print(f'test r2 = {test_metrics.r2()}')
print(f'test r2_score = {test_metrics.r2_score()}')

# %%

plot_ci(test_predicted, alpha, type='test')
plt.tight_layout()
plt.show()

# %%

plot_ci(test_predicted.iloc[0:50], alpha, type='test')
plt.tight_layout()
plt.show()



