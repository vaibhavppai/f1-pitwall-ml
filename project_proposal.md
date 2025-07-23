<h1 align="center">Formula 1 Race Predictor</h1>
<h2 align="center">Project Proposal</h2>
<h3 align="center">CS 7641 – Summer 2025 – Group 4</h3>

---

## Introduction / Background

Formula 1 is the most popular motorsport in the world, attracting millions of viewers and offering large prize pools. As such, identifying competitive advantages and predicting performance are vital for influencing race outcomes. Brusik [1] compares a wide range of univariate time-forecasting models along with more complex multivariate deep learning techniques to predict race time, demonstrating that multivariate models perform more accurately. Haber et al. [2] dives into these numerous factors to predict race winners, using Random Forests with SHAP analysis to identify key features. Finally, Menon et al. [3] uses Monte Carlo analysis to predict outcomes and determine how variance in race results can be explained.

A few datasets will be combined for analysis. **[FastF1](https://docs.fastf1.dev/)** is a python package for accessing and analyzing Formula 1 results, schedules, timing data, and telemetry. **[TUMFTM Racetrack Database](https://github.com/TUMFTM/racetrack-database)** is a repository consisting of centerlines, track widths and race lines for F1 racetracks. Finally, **[Tomo Bacinger’s F1 Circuit Data](https://github.com/bacinger/f1-circuits)** is a repository of F1 circuits data in GeoJSON format. We will pull data and generate features from these sources. 

---

## Problem Definition and Motivation

Formula 1 is a data-dependent sport where success is dictated by a combination of driver skills, engineering, track conditions, and strategy. Each race results from multiple interdependent variables, making it an ideal candidate for machine learning-based predictions.  

The motivation behind developing such predictive tools is threefold: 

### Race Strategy Simulation 
Decisions like when to pit, whether to undercut a competitor, or how aggressively to push a tire compound are made in real time under intense pressure. A robust prediction framework trained on historical data can serve as a surrogate for simulating race scenarios. 

### Sports Analytics and Research 
F1 presents a unique challenge: it is a small-data, high-complexity environment with structured multivariate data heavily dependent on time and has non-stationary patterns due to rule changes, car upgrades, and track changes. Developing effective predictive models enables exploration of critical research topics such as: 

- Multimodal feature integration 

- Time-series forecasting in sparse regimes 

- Interpretable model design in real-world systems 

### Fan and Betting Communities 
Publicly available predictions are often heuristic-based or overly simplistic. We seek to make accurate, data-driven predictions accessible and interpretable to the wider F1 audience. 

---

## Methods

### Data preprocessing 

- **Data Integration** - combine race data, weather conditions, and driver/team stats from multiple sources. 

- **Feature Engineering** – extract relevant race and driver specific attributes (eg. tire age, sector deltas, weather) that significantly influence predictions. 

- **Categorical Encoding** – convert non-numerical data like driver names and tire compounds into numerical values. 

- **Normalization/Scaling** – ensure features like lap time, speed, and tire wear are on comparable scales to improve model convergence and performance. 

### Unsupervised learning 

- **PCA** - reduce high-dimensional race data (eg. lap-by-lap telemetry) into key components for faster, more interpretable analysis. 

- **K-Means** - group races or stints with similar tire degradation or strategy  

- **DBSCAN** - detect anomalies or outlier laps affected by pit-stops, red flags or safety cars. 

### Supervised Learning 

- **Random Forest, Support Vector Machine, Gaussian Naive Bayes, K-Nearest Neighbors, XGBoost, ANN** - predict outcomes like finishing positions, lap times, tire strategy, and fastest lap categories. 

---

## (Potential) Results and Discussion

### Metrics 
We will evaluate our models using **Accuracy, Precision, Recall, F1-score** (for classification), and **MSE, R², Cross-Validation** (for regression and model robustness). 

### Project Goals 
We aim to make predictions for pit-stop strategies, classify driver styles, predict effects of weather, track-specific analyses and estimate optimal racing lines tailored to qualifying and race sessions. We use publicly available data and aim to create models that are lightweight. 

### Expected Results 
We aim for **F1-score > 0.85** and **R² > 0.90**, with interpretable clusters and strategy insights supporting real-time race analysis. 

---

**Word Count** - 597

---

## References

[1] F. Brusik, Predicting Lap Times in a Formula 1 Race Using Deep Learning Algorithms: A Comparison of Univariate and Multivariate Time Series Models, M.S. thesis, School of Humanities and Digital Sciences, Tilburg University, 2024. 

[2] E. El Haber, E. Sawaya, M. Attieh, A. Tannous, W. Ghazaly, and M. Owayjan, "Formula 1 Race Winner Prediction Using Random Forest and SHAP Analysis," in 2025 International Conference on Control, Automation, and Instrumentation (IC2AI), 2025, pp. 1270-1274. 

[3] S. A. Menon, M. K. Ranjan, A. Kumar, and B. Gopalsamy, "F1 Lap Analysis and Result Prediction," International Research Journal of Modernization in Engineering Technology and Science, vol. 6, no. 11, Nov. 2024. 

[4] C. L. W. Choo, Real-time decision making in motorsports: analytics for improving professional car race strategy, Ph.D. dissertation, Massachusetts Institute of Technology, 2015. 

[5] M. Keertish Kumar and N. Preethi, "Formula One Race Analysis Using Machine Learning," in Proc. 3rd Int. Conf. Recent Trends in Machine Learning, IoT, Smart Cities and Applications (ICMISC), Singapore, 2023, pp. 533-540, Springer Nature Singapore. 

---

## Gantt Chart

View the [Gant Chart](https://gtvault.sharepoint.com/:x:/s/ML7641Project568/EUXYbfK5vtJGrfBQwmEdczsBMaH1GnclwjVkn6ma7PoFgg?e=tfWXDv)

---

## Contribution Table

| Team Member           | Proposal Contributions                             |
|-----------------------|--------------------------------------------|
| Krtin Kala            | Problem Definition, Contribution Table    |
| Vaibhav Pai           | Gantt Chart                                |
| Paras Singh           | GitHub Repository, Methods, Results        |
| Carter Tegen          | Documentation Layout, Problem Definition   |
| Sreevickrant Sreekanth| Presentation, Problem Definition           |

---

## Video Presentation

View our [Project Proposal Presentation Video](https://youtu.be/QrAcd8dy-o4)

---

## Project Award Eligibility

We will opt in for the award consideration.

