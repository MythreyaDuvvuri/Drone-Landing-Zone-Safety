# Drone-Landing-Zone-Safety-
Drone Landing Zone Safety Prediction Using Machine Learning
A Logistic Regression model to predict whether a zone is safe to land the drone. It uses terrain features derived from aerial imagery to support drone landing operations. 

Repository Structure
drone_landing_log_reg_predictor.pkl - The Logistic Regression model used to predict
drone_landing_log_reg_scaler.pkl - The fitted scaler to normalize feature values
Landing Zone Dataset - Sheet1.csv - The data on which the model is trained on
Logistic Regression Test Metrics.csv - The test metrics of the model
Logistic Regression ROC.png - The ROC curve of the model
Pred Heatmap.png - The heatmap of predictions on test split
True Heatmap.png - The heatmap of true labels of test split
Exploration.ipynb - The notebook containing data exploration on the dataset
Model validating.ipynb - The notebook containing model validation and training
landingPredicting.py - A script containing helpful functions
Explanation video.mp4 - Video showing the files included and the process used to train the model
README.txt - This document

Feature Schema
In order:
slope_deg(float): slope of the surface to land
roughness(float): texture of the surface. 0 means perfectly smooth 
edge_density(float): represents density of sharp edges in the image. 0 means no edges detected
ndvi_mean(float): represents the ndvi value used to identify foliage/vegetation in image. 0 means no foliage detected
shadow_fraction(float): shadows present in image. Shadows in the image can cause detecting color and edges harder. 0 means no shadows
brightness_std(float): the standard deviation of brightness in the image. It represents different materials having different brightness. 
0 means the zone is of same material uniformly. Higher values indicate the floor is made of multiple materials 
object_density(float): number of objects present in image. if there are multiple objects, landing may become unsafe
confidence_score(float): confidence in detecting the features from image. Higher confidence score means the other feature values are more reliable

Preprocessing:
drone_landing_log_reg_scaler.pkl is used to normalize the data

The helper module landingPredicting.py contains these functions:
get_safe_input_ranges - returns a dictionary containing the feature names and their min/max safe ranges to use while predicting
predict_single - takes a single datapoint containing the features and predicts using the model
predict_grid - takes an array of data and the grid size(dimensions) as tuple to predict and generate a heatmap of safe landing zones. 
		returns an array of the given dimensions with predictions. 
		generates and saves a heatmap image in .png format

Install these packages to ensure dependencies:
pip install numpy pandas scikit-learn matplotlib joblib

Sample code:

import numpy as np
import pandas as pd
import import landingPredicting

df = pd.read_csv("data_to_predict_on.csv")

get_safe_input_ranges()

X = df.iloc[[42]]
pred = predict_single(X)

X_grid = df.iloc[51:91]
grid_shape = 8, 5
preds_grid = predict_grid(X_grid, grid_shape)

import joblib
model_name = "drone_landing_log_reg_predictor.pkl"
scaler_name = "drone_landing_log_reg_scaler.pkl"
model = joblib.load(model_name)
scaler = joblib.load(scaler_name)

values = df.iloc[51:91]
values_scaled = scaler.transform(values)
preds = model.predict(values_scaled)

For accurate predictions, ensure the data is in the right schema and is within the safe ranges


Author and Contact Info
Developed by Mythreya Duvvuri
Contact: dmythreya@gmail.com
Date: 14 July 2026
