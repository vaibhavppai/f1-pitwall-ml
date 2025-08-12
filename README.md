## Formula 1 AI PitWall

### CS 7641 - Machine Learning - Group 4 Project

**Repository for the Formula 1 AI PitWall project.** This document provides a detailed overview of the project structure, explaining the purpose of all relevant directories and files

---

## Table of Contents
1.  [Project Overview](#-project-overview)
2.  [Repository Structure](#-repository-structure)
    - [Root Directory](#root-directory)
    - [`Qualifying_Lap_Predictor/`](#qualifying_lap_predictor)
    - [`pit_stop_predictor/`](#pit_stop_predictor)
    - [`track_clustering/`](#track_clustering)
    - [`Tire_Analysis/`](#tire_analysis)
    - [`driver-analysis/`](#driver-analysis)
3.  [The Team](#-the-pit-crew)

---

## Project Overview

This project is a multi-faceted machine learning analysis designed to decode the DNA of a Formula 1 race weekend. It is built on five distinct analytical models, each deconstructing a key component of a race:

* **Driver Performance & Style Analysis:** A deep dive into individual driver performance using telemetry data to identify unique braking signatures, cornering styles, and data-driven driver archetypes
* **Tire Choice Modeling:** Analyzes historical weather data to predict lap time improvements and optimal tire compound choices
* **Qualifying Pace Predictor:** Forecasts the ultimate single-lap pace of each driver using a wide range of performance and environmental metrics
* **Track DNA Analysis:** Uses unsupervised clustering to categorize every F1 circuit into distinct, data-driven archetypes
* **Pit Stop & Stint Predictor:** Employs time-series analysis with an LSTM to predict tire degradation and the remaining viable laps in a stint

---

## Repository Structure

This repository is organized into directories for each sub-project, along with the source code for the project website.

### Root Directory

Contains the website's HTML pages and project configuration files.

* `index.html`: The main landing page for the project website
* `driver-analysis.html`: The webpage detailing the driver analysis
* `qualifying-pace-predictor.html`: The webpage detailing the qualifying predictor analysis
* `pit-stop-predictor.html`: The webpage for the pit stop predictor analysis
* `track-analysis.html`: The webpage for the track clustering analysis
* `tire-choice-model.html`: The webpage for the tire choice modeling analysis
* `/assets`: Contains all static files for the website, including `/css` and `/js`
* `.gitignore`: Specifies which files and directories to exclude from Git version control
* `README.md`: This file


### `driver-analysis/`

Contains a detailed analysis of individual driver styles and performance characteristics based on telemetry data.

* `corner_clusters_[track_name].png`: A series of plots visualizing cornering behavior clusters for drivers at specific circuits (e.g., Silverstone, Monaco, Spa)
* `driver_archetypes.png`: A visualization classifying different drivers into data-driven archetypes based on their driving style
* `braking_sig.png`: A plot identifying unique braking signatures among drivers
* `lec_quadrant.png` / `ric_quadrant.png`: Specific case-study plots for individual drivers (Charles Leclerc and Daniel Ricciardo)
* `telemetry_anomaly.png`: A plot showing detected anomalies in driver telemetry data


### `Qualifying_Lap_Predictor/`

Contains the complete end-to-end pipeline for the qualifying lap time prediction model

* **`/data_collection`**: Scripts and notebooks for gathering and merging raw data
    * `/FP2_data`: Contains scripts to collect Free Practice 2 data, organized by year
    * `/Quali_data`: Contains the notebook to collect Qualifying session data
    * `/Combined_Quali_FP2_data`: Contains the `merging_data.ipynb` notebook to combine the FP2 and Qualifying datasets
* **`/feature_engineering`**: The `feature_engineering.ipynb` notebook processes the combined raw data, creating the final, clean dataset (`f1_processed_qualifying_data_with_FP2.xlsx`) for modeling
* **`/model_training/Final`**: Contains the final Jupyter Notebooks (`model_training.ipynb`, `model_training_with_SVR_and_Lasso.ipynb`) used to train, tune, and evaluate the various regression models
* **`/images`**: Holds all output plots and visualizations, such as prediction comparisons and feature correlation maps
* 
### `pit_stop_predictor/`

Contains the pipeline for the tire stint and degradation prediction models.

* **`/data`**: Stores the raw data, organized by year (2021-2025). Each year's folder contains round-by-round CSV files for both lap-by-lap data (`..._laps.csv`) and weather conditions (`..._weather.csv`)
* **`/eda`**: The `tire_eda.ipynb` notebook is used for initial exploratory data analysis and preprocessing of the raw data
* **`/learning`**: Contains the initial modeling approach (Random Forest, XGBoost), including notebooks and saved `.joblib` model pipelines
* **`/strat2`**: Holds the final, more advanced time-series modeling strategy
    * `training_lstm.ipynb`: The notebook used for training the final LSTM model
    * `tire_stint_model_pytorch.pth`: The saved, pre-trained PyTorch model file for the LSTM
* **`/img`**: Stores all visualizations generated during the analysis

### `track_clustering/`

Contains the code and data for the unsupervised learning task of classifying F1 circuits

* **`/data`**: Contains the raw `f1-circuits-2022-2025.geojson` file with track geometry cleaned from the original databse
* **`/source`**: The core Python scripts (`dbscan_correct.py`, `gmm.py`, `curvature geojson gmm 3d with 4 categories.py`) that implement the feature engineering and clustering algorithms
* **`/plots`**: Contains all output visualizations of the track clusters

### `Tire_Analysis/`

A focused analysis on tire compounds and performance

* **`data_analysis.ipynb`**: The main Jupyter Notebook where the regression and classification analyses are performed
* **`/images`**: Holds all generated plots for this section, such as the confusion matrix and hyperparameter sweep results


## The Pit Crew

* **Krtin Kala:** Track Clustering, Driver Analysis, Web Page Design
* **Vaibhav Pai:** Track Clustering, Driver Analysis, Web Page Design
* **Paras Singh:** Best Qualifying Lap Predictor
* **Carter Tegen:** Tire Change Analysis
* **Sreevickrant Sreekanth:** Tyre Stint and Pitstop Predictor
