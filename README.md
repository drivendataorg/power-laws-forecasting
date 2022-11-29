[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://s3.amazonaws.com/drivendata-public-assets/se-challenge-1-banner.jpg)

#  Power Laws: Forecasting Energy Consumption
## Goal of the Competition

Building energy forecasting has gained momentum with the increase of building energy efficiency research and solution development. Indeed, forecasting the global energy consumption of a building can play a pivotal role in the operations of the building. It provides an initial check for facility managers and building automation systems to mark any discrepancy between expected and actual energy use. Accurate energy consumption forecasts are also used by facility managers, utility companies and building commissioning projects to implement energy-saving policies and optimize the operations of chillers, boilers and energy storage systems.

Planning and forecasting the use of the electrical energy is the backbone of effective operations. Energy demand forecasting is used within multiple Schneider Electric offers, and different methods require more or less data. Schneider Electric is interested in more precise and robust forecasting methods that do well with little data.

** In [this competition](https://www.drivendata.org/competitions/51/electricity-prediction-machine-learning) data scientists all over the world built algorithms to forecast building consumption reliably. **

The best algorithms were generally ensembles of models with modelers that thought very, very carefully about how to avoid using future data in their timeseries models. Using weather, holidays, synthetic features, and LightGBM, these participants created the best predictions!

## What's in this Repository
This repository contains code from winning competitors in the [Power Laws: Forecasting Energy Consumption](https://www.drivendata.org/competitions/51/electricity-prediction-machine-learning/) DrivenData challenge.

#### Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).


## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | --- | --- | ---
1 | slonoslon | 0.001369    | 0.001956   | First, I did a lot of preprocessing (history-based aggregate functions calculation) of original data to produce features for each point in train and test sets. Then, these features are used to train gradient boosting (LightGBM) model.
2 | jacquespeeters | 0.001218    | 0.001969 | The final model is a simple average of 3 LightGbm models with different set of parameters (number of leaves). The mdoels have a time series specific training process: train and validate models with time split in order to avoid any leak (in my case determine the number of iterations), then re-train on full data before predicting. In the end, there is 6 models trained (3 set of parameters, once on train/valid, once on full data) on all 300 Building/SiteId. It takes around 2 hours to train.
3 | willkoehrsen | 0.001285 | 0.002039 | My approach treats the problem as a standard supervised regression task. Given a set of features – the time and weather information – we want to build a model that can predict the continuous target, energy consumption. The model is trained on the past historical energy consumption using the features and the target and then can be used to make predictions for future dates where only the features are known.  The final machine learning model I built is an ensemble of two ensemble methods (a meta-ensemble I guess you could call it), the random forest and the extra trees algorithms. The final model is made up of six random forest regressors and six extra trees regressors, each of which uses a different number of decision trees (from 50 to 175 spaced by 25). (All of the algorithms are implemented using the Scikit- learn Python library). The final predictions are made by averaging the results of these 12 individual estimators.

#### [Interview with winners](http://drivendata.co/blog//)

