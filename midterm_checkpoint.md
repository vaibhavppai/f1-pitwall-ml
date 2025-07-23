
<h1 align="center">Formula 1 Race Predictor</h1>
<h2 align="center">Midterm Checkpoint</h2>
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

#### Tire Choice Modeling (Carter)

### Sports Analytics and Research 
F1 presents a unique challenge: it is a small-data, high-complexity environment with structured multivariate data heavily dependent on time and has non-stationary patterns due to rule changes, car upgrades, and track changes. Developing effective predictive models enables exploration of critical research topics such as: 

- Multimodal feature integration 

- Time-series forecasting in sparse regimes 

- Interpretable model design in real-world systems 

### Fan and Betting Communities 
Publicly available predictions are often heuristic-based or overly simplistic. We seek to make accurate, data-driven predictions accessible and interpretable to the wider F1 audience. 



---

## Methods

### Tire Choice Modeling (Carter)

#### Data Proprocessing
The data was downloaded via the FastF1 database. The main preprocessing technique in the current work is data reduction via feature selection. For regression analysis of predicting lap time improvement by switching tires, the chosen features are reduced to Air Temperature, Track Temperature, Humidity, Pressure, Rainfall (T/F), Wind Direction, and Wind Speed. The chosen model, XGBoost, is generally pretty resiliant to poorly transformed data since it is tree based, but there is certainly room for future work to help aid the convergence of these models.

#### ML Algorithms/Models Implemented
The chosen model for the classification and regression problems was XGBoost. It was chosen because it is a well performing tree based model for both of these problems, which performs very well at scale and can eb easily tuned to prevent overfitting. It's a great starting point due to it's simplicity, but simpler models shall be evaluated for the final report. Additionally, unsupervised models such as clustering will be implemented on this problem in the final report to see if there are groupings between weather conditions that may reveal trends in tire selection, before looking at lap time improvements.

#### Unsupervised and Supervised Learning Method Implemented
---

## Results and Discussion

### Tire Choice Modeling (Carter)
#### Visualizations
#### Quantitative Metrics
#### Analysis of Model
#### Next Steps

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
