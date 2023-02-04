"""
==============
1. EDA
==============
"""

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import pandas as pd

from ai4water.eda import EDA
from easy_mpl import plot, boxplot, hist
from easy_mpl.utils import create_subplots

from utils import make_data,  box_violin

# %%
# Loading the original dataset

data, _, _, _ = make_data()


# %%
# Here, we are printing the shape of original dataset.
# The first value shows the number of samples/examples/datapoints
# and the second one shows the number of features.

print(data.shape)

# %%
# The first five samples are

data.head()

# %%
# The last five samples are

data.tail()

# %%
# The names of different adsorbents and their counts

data['Adsorbent'].value_counts()

# %%
# The names of different Feedstock and their counts

data['Feedstock'].value_counts()

# %%
# The names of different Anion_type and their counts

data['Anion_type'].value_counts()

# %%
# Removing the categorical features from our dataframe

data.pop("Adsorbent")
data.pop("Feedstock")
data.pop("Anion_type")

# %%
# get statistical summary of data

pd.set_option('display.max_columns', None)

print(data.describe())

# %%
# initializing an instance of EDA class from AI4Water
# in order to get some insights of the data

eda = EDA(data = data, save=False, show=False)

# %%
# plot correlation between numerical features

ax = eda.correlation(figsize=(12,10))
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, weight='bold')
plt.tight_layout()
plt.show()

# %%
# making a line plot for numerical features

fig, axes = create_subplots(data.shape[1])

for ax, col, label  in zip(axes.flat, data, data.columns):

    plot(data[col].values, ax=ax, ax_kws=dict(ylabel=col),
         lw=0.9,
         color='darkcyan', show=False)
plt.tight_layout()
plt.show()

# %%

fig, axes = create_subplots(data.shape[1])
for ax, col in zip(axes.flat, data.columns):
    boxplot(data[col].values, ax=ax, vert=False, fill_color='lightpink',
            flierprops={"ms": 1.0}, show=False, patch_artist=True,
            widths=0.6, medianprops={"color": "gray"},
            ax_kws=dict(xlabel=col, xlabel_kws={'weight': "bold"}))
plt.tight_layout()
plt.show()

# %%
# show the box and (half) violin plots together

fig, axes = create_subplots(data.shape[1])
for ax, col in zip(axes.flat, data.columns):
    box_violin(ax=ax, data=data[col], palette="Set2")
    ax.set_xlabel(xlabel=col, weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.tight_layout()
plt.show()


