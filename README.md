<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Credit card fraud detection using Neural networks</div>
<div align="center"><img src="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/intro.gif?raw=true"></div>



## Overview:
Our objective is to create the classifier for credit card fraud detection. To do it, we'll compare classification models from different methods :

- Logistic regression
- Support Vector Machine
- Bagging (Random Forest)
- Boosting (XGBoost)
- Neural Network (tensorflow/keras)
## Dataset:
[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. I decided to proceed to an undersampling strategy to re-balance the class.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data.
## Implementation:

**Libraries:**  `NumPy` `pandas` `pylab` `matplotlib` `sklearn` `seaborn` `plotly` `tensorflow` `keras` `imblearn`
## Data Exploration:
Only 492 (or 0.172%) of transaction are fraudulent. That means the data is highly unbalanced with respect with target variable Class.
<img src ="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/eda1.PNG?raw=true">
<br>
The dataset is highly imbalanced ! It's a big problem because classifiers will always predict the most common class without performing any analysis of the features and it will have a high accuracy rate, obviously not the correct one. To change that, I will proceed to random undersampling.

The simplest undersampling technique involves randomly selecting examples from the majority class and deleting them from the training dataset. This is referred to as random undersampling.

Although simple and effective, a limitation of this technique is that examples are removed without any concern for how useful or important they might be in determining the decision boundary between the classes. This means it is possible, or even likely, that useful information will be deleted.

<img src="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/undersampling.PNG?raw=true">
<br>
For undersampling, we can use the package imblearn with RandomUnderSampler function.<br>
```
import imblearn
from imblearn.under_sampling import RandomUnderSampler 
undersample = RandomUnderSampler(sampling_strategy=0.5)
```

<img src= "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/eda2.PNG?raw=true">

## Machine Learning Model Evaluation and Prediction:
### Logistic Regression:
<img src ="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/lr.PNG?raw=true" width="33%"> <img src= "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/lr1.PNG?raw=true" width="33%"> <img src="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/lr2.PNG?raw=true" width="33%">
```
Accuracy : 0.94
F1 score : 0.92
AUC : 0.96
```

### Support Vector Machine:
<img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/svm.PNG?raw=true" width="33%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/svm2.PNG?raw=true" width="33%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/svm1.PNG?raw=true" width="33%">
```
Accuracy : 0.94
F1 score : 0.92
AUC : 0.97
```


### Random Forest:
<img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/RF.PNG?raw=true" width="33%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/RF1.PNG?raw=true" width="33%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/RF2.PNG?raw=true" width="33%">
```
Accuracy : 0.95
F1 score : 0.93
AUC : 0.97
```

### XGBoost:
<img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/XG.PNG?raw=true" width="33%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/XG1.PNG?raw=true" width="33%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/XG2.PNG?raw=true" width="33%">
```
Accuracy : 0.95
F1 score : 0.93
AUC : 0.97
```
### Multi Layer Perceptron:
<img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/percept.PNG?raw=true" width="33%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/percept1.PNG?raw=true" width="33%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/percept2.PNG?raw=true" width="33%">
```
Accuracy : 0.95
F1 score : 0.94
AUC : 0.98
```
### Neural Networks:
<img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/NN1.PNG?raw=true" width="40%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/NN2.PNG?raw=true" width="40%">
<img src="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/ANN.PNG?raw=true" width="33%"> <img src ="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/ANN1.PNG?raw=true" width="33%"> <img src="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/ANN2.PNG?raw=true" width="33%">
```
Accuracy : 0.95
F1 score : 0.94
AUC : 0.98
```





### Lessons Learned
`Classification Algorithms`
`Multilayer Perceptrons`
`XGBoost classifier`
`Bagging`
`Boosting`







## Related:
[Credit card fraud detection using Ensemble methods](https://github.com/Pradnya1208/Credit-card-fraud-detection-using-ensemble-learning-predictive-models)
[Credit card fraud detection using Isolation Forest and LOF](https://github.com/Pradnya1208/Credit-card-fraud-detection-using-Isolation-Forest-and-LOF)
### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner








[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

