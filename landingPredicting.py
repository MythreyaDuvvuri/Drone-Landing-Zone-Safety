#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn as sk
import joblib
import matplotlib.pyplot as plt


# In[6]:


model_name = "drone_landing_log_reg_predictor.pkl"
scaler_name = "drone_landing_log_reg_scaler.pkl"
model = joblib.load(model_name)
scaler = joblib.load(scaler_name)


# In[ ]:


def predict_grid(grid_values, grid_shape):
    # Returns a grid of values of 0 and 1. Generates a heatmap and saves it
    # Give grid_values in the same format of model
    # grid_shape must be a tuple and representing the 2D shape of the grid
    grid_values_scaled = scaler.transform(grid_values)
    y_preds = model.predict(grid_values_scaled)
    y_pred_grid = y_preds.reshape(grid_shape)
    for x in range(y_pred_grid.shape[0]):
        for y in range(y_pred_grid.shape[1]):
            if y_pred_grid[x][y] == 0:
                res = "rs"
            else:
                res = "gs"
            plt.plot(x, y, res)


    plt.title(label="Predicted Heatmap")
    plt.savefig("Predicted Grid Heatmap.png")
    print("Heatmap saved to Predicted Grid Heatmap.png")
    return y_pred_grid



# In[ ]:


def predict_single(values):
    # Returns a single 0 or 1 to predict unsafe or safe
    values_scaled = scaler.transform(values)
    y_pred = model.predict(values_scaled)
    print(f"{"Safe" if y_pred[0] == 1 else "Unsafe"}")
    return y_pred


# In[30]:


np.set_printoptions(suppress=True)
ranges = np.array([scaler.mean_ - (3 * np.sqrt(scaler.var_)), scaler.mean_ + (3 * np.sqrt(scaler.var_))])

ranges_dict = {
    "feature_names" : scaler.feature_names_in_,
    "min" : [], 
    "max" : []
}
for x in range(len(ranges[0])):
    if ranges[0][x] < 0:
        ranges[0][x] = 0

    ranges_dict["min"].append(ranges[0][x])

    if x == 0:
        ranges_dict["max"].append(ranges[1][x])
        continue

    if ranges[1][x] > 1:
        ranges[1][x] = 1

    ranges_dict["max"].append(ranges[1][x])

ranges_dict["min"] = np.array(ranges_dict["min"])
ranges_dict["max"] = np.array(ranges_dict["max"])


def get_safe_input_ranges():
    # Returns a dict containing the feature names, their safe min and max limits
    return ranges_dict

