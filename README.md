 # Spaceship Titanic   ![icons8-rocket-48](https://github.com/TOFWAQ/Kaggle_Challenge/assets/37884436/bbf997e5-9c67-4f71-b338-9946f6d62aa8)


The spaceship Titanic is about a fictional mystery where a spaceship collides with an anomaly, resulting in the disappearance of half of the passengers on board. We aim to help rescue the crew and passengers by predicting which passengers were transported by the anomaly based on records recovered from the spaceship’s damaged computer system.

<img width="270" alt="Picture1" src="https://github.com/TOFWAQ/Kaggle_Challenge/assets/37884436/c4340d72-7a6f-4e87-a499-f40e68ccc667">

## Tools
- Goggle Colab Notebook,
- Python 3.7+


## File Description

- train.csv : Personal records for about two-thirds (~8700) of the passengers, to be used as training data.

- test.csv : Personal records for the remaining one-third (~4300) of the passengers, to be used as test data.

- sample_submission.csv : A submission file in the correct format

## Import Packages
```ruby
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import OrdinalEncoder
!pip install fancyimpute
from fancyimpute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
!pip install --upgrade scikit-learn
from sklearn.metrics import confusion_matrix
```
## Analysis
- we load the train dataset
```ruby
train_link = ('https://raw.githubusercontent.com/TOFWAQ/Train_data/main/train.csv')
train = pd.read_csv(train_link)
train.head()
```
<img width="832" alt="image" src="https://github.com/TOFWAQ/Kaggle_Challenge/assets/37884436/98777cd3-3886-4e77-ae2b-43d83ac8f02b">


- Age Distribuition of Passengers in train data
![download (8)](https://github.com/TOFWAQ/Kaggle_Challenge/assets/37884436/a515a0ee-44ed-4e75-aa62-042427d867d8)

- Distribution of Passengers Across HomePlanet in train data
![download (9)](https://github.com/TOFWAQ/Kaggle_Challenge/assets/37884436/d81d8157-b0e4-47e4-81e3-ed12def222a6)

### Model Preprocessing
```ruby
# Initialize the label encoder
le = LabelEncoder()

# Label encode to assign values to similar strings
train['HomePlanet'] = le.fit_transform(train['HomePlanet'])
train['Destination'] = le.fit_transform(train['Destination'])
train['Transported'] = le.fit_transform(train['Transported'])
train['CryoSleep'] = le.fit_transform(train['CryoSleep'])
train['VIP'] = le.fit_transform(train['VIP'])
```
```ruby
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Create an imputer object that replaces NaN values with the mean value of the column
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform it
x_train_imputed = imputer.fit_transform(train_n)

# Convert imputed output to a pandas dataframe
train_n = pd.DataFrame(x_train_imputed, columns=train_n.columns)
train_n.head()
```

### Model Evaluation

#### Draw Confusion Matrix

```ruby
con_m = confusion_matrix(y_test, y_predic)


#create the label
labels = ['TN', 'FP', 'FN', 'TP']
categories = np.asarray(labels).reshape(2,2)

#combine the label and values
# Create a new array for annotations, combining labels and values
annotations = [f"{label}\n{value}" for label, value in zip(labels, con_m.flatten())]
annotations = np.asarray(annotations).reshape(2,2)
# Visualize the confusion matrix
sns.heatmap(con_m,annot=annotations,fmt='', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```
![confusion](https://github.com/TOFWAQ/Kaggle_Challenge/assets/37884436/d61a82dc-1ab4-4493-bd6d-805dc6e6b760)

#### Visualise the ROC Curve
```ruby
from sklearn.metrics import roc_curve, auc
# Plot the ROC curve
#Compute the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_predic)

# Compute the AUC
roc_auc = auc(fpr, tpr)
# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='navy', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC GradientBoosting')
plt.legend(loc="lower right")
plt.show()
```

![gradient](https://github.com/TOFWAQ/Kaggle_Challenge/assets/37884436/5020a1ed-20df-4289-8c69-2f252eeb2fdb)



## Authors
- [@Oshgig](https://github.com/Oshgig)
- [@titobi](https://github.com/titobi)
- [@Damilolaori](https://github.com/Damilolaori)
- [@Wolexide](https://github.com/Wolexide)
- [@franchaise](https://github.com/franchaise)
- [@ebereinyiama](https://www.github.com/ebereinyiama)

### Kaggle Kernel [SpaceshipTitanic](https://www.kaggle.com/competitions/spaceship-titanic/data)
