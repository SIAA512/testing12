"""
====================================================
4. Aleatoric Bayesian neural network (aleatoric_BNN)
====================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from SeqMetrics import RegressionMetrics

from utils import  get_dataset, \
    make_data, run_experiment, \
    compute_predictions, create_aleatoric_bnn_model, \
    plot_ci, negative_loglikelihood

# %%

dataset , *_ = get_dataset(encoding="le")
X_train, y_train = dataset.training_data()
X_test, y_test = dataset.test_data()

# %%

hidden_units = [8, 8]
learning_rate = 0.006
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

aleatoric_bnn_model = create_aleatoric_bnn_model(len(y_train)+len(y_test),
                                                 hidden_units)

# %%

run_experiment(aleatoric_bnn_model, negative_loglikelihood, train_dataset,
               test_dataset, learning_rate, num_epochs)

# %%

tr_examples, tr_targets = list(train_dataset.unbatch().
                               shuffle(batch_size * 10).batch(len(y_train)))[0]

# %%

test_examples, test_targets = list(test_dataset.unbatch().
                                   shuffle(batch_size * 10).batch(len(y_test)))[0]

# %%
# Training

tr_prediction_distribution = aleatoric_bnn_model(tr_examples)
tr_prediction_mean = tr_prediction_distribution.mean().numpy()
tr_prediction_stdv = tr_prediction_distribution.stddev().numpy()

# The 95% CI is computed as mean ± (1.96 * stdv)
upper = (tr_prediction_mean.tolist() + (1.96 * tr_prediction_stdv)).tolist()
lower = (tr_prediction_mean.tolist() - (1.96 * tr_prediction_stdv)).tolist()
tr_prediction_stdv = tr_prediction_stdv.tolist()

# %%

tr_metrics = RegressionMetrics(tr_targets.numpy(),
                               np.array(tr_prediction_mean))
print(f'train r2 = {tr_metrics.r2()}')
print(f'train r2_score = {tr_metrics.r2_score()}')

# %%

tr_df = pd.DataFrame()
tr_df['prediction'] = tr_prediction_mean.reshape(-1,)
tr_df['upper'] = np.array([val[0] for val in upper])
tr_df['lower'] = np.array([val[0] for val in lower])

plot_ci(tr_df, alpha, type='train')
plt.tight_layout()
plt.show()

# %%

plot_ci(tr_df.iloc[0:50], alpha, type='train')
plt.tight_layout()
plt.show()

# %%
# Test

test_prediction_distribution = aleatoric_bnn_model(test_examples)
test_prediction_mean = test_prediction_distribution.mean().numpy()
test_prediction_stdv = test_prediction_distribution.stddev().numpy()

# The 95% CI is computed as mean ± (1.96 * stdv)
upper = (test_prediction_mean.tolist() + (1.96 * test_prediction_stdv))\
    .tolist()
lower = (test_prediction_mean.tolist() - (1.96 * test_prediction_stdv))\
    .tolist()
test_prediction_stdv = test_prediction_stdv.tolist()

# %%

test_metrics = RegressionMetrics(test_targets.numpy(),
                               np.array(test_prediction_mean))
print(f'test r2 = {test_metrics.r2()}')
print(f'test r2_score = {test_metrics.r2_score()}')

# %%

test_df = pd.DataFrame()
test_df['prediction'] = test_prediction_mean.reshape(-1,)
test_df['upper'] = np.array([val[0] for val in upper])
test_df['lower'] = np.array([val[0] for val in lower])

plot_ci(test_df, alpha, type='test')
plt.tight_layout()
plt.show()

# %%

plot_ci(test_df.iloc[0:50], alpha, type='test')
plt.tight_layout()
plt.show()
