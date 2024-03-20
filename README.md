# Spaceship Titanic
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


#### Age Distribuition of Passengers
![download (8)](https://github.com/TOFWAQ/Kaggle_Challenge/assets/37884436/a515a0ee-44ed-4e75-aa62-042427d867d8)

#### Distribution of Passengers Across HomePlanet
![download (9)](https://github.com/TOFWAQ/Kaggle_Challenge/assets/37884436/d81d8157-b0e4-47e4-81e3-ed12def222a6)



## Authors
- [@Oshgig](https://github.com/Oshgig)
- [@titobi](https://github.com/titobi)
- [@Damilolaori](https://github.com/Damilolaori)
- [@Wolexide](https://github.com/Wolexide)
- [@octokatherine](https://www.github.com/octokatherine)
- [@ebereinyiama](https://www.github.com/ebereinyiama)

### Kaggle Data [SpaceshipTitanic](https://www.kaggle.com/competitions/spaceship-titanic/data)
