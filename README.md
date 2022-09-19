# Email Spam Classification



<img src="/images/email-spam.png" align="left" width="200" />

## Goal

The aim of this project is to classify emails into spam and ham emails.  
To do this I will use the frequencies method that will count  how many different words there are in a sentence after that I will choose a number of relevant words.Below I will expalin better the method

<br clear="left"/>

[Go to Conclusion](#Conclusion)

## Libraries used

```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly_express as px
import plotly.figure_factory as ff
from wordcloud import WordCloud
import nltk

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn import metrics

import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

```

## Loading Dataset

Dataset has been downloaded from kaggle.You can see more info clicking **[here](https://www.kaggle.com/code/mfaisalqureshi/email-spam-detection-98-accuracy/data)**
```
print(data.shape)
(5572, 2)
```

## EDA
I am going to obtain a general overview of the dataset using graphs and adding also a new feature length that describes the number of characters of each email

![](/images/counttarget.png)

The dataset is unbalanced, we can observe that there are:
- 747 observations as Spam
- 4825 observations as ham

```
# Adding new feature

#apply len columns to entire dataset and for train and test dataset
data['length'] = data['Message'].apply(len)
```
Now I  will observe if there are difference in length between email spam and not

![](/images/distribution%20length.png)

From the above distributions plot we can observe that length is a powerful feature since is able to provide an useful information

# Feature Engineering

## Preprocess the data

In this phase first I will perform the following steps:
- convert tha label feature in numerical feature 
- convert web addresses into abbreviation 'wb'
- convert  phone numbers into abbreviation 'pn'
- convert  numbers into abbreviation 'p'
-  encode  money symbols into abbreviation 'ms'
- remove punctuation and white spaces
- convert all text to lowercase

### Removing Stopwords

Going forward, I'll remove stopwords from the message content. Stop words are words that search engines have been programmed to ignore, such as “the”, “a”, “an”, “in”, "but", "because" etc.,we can resume stopwords as words that don't have a meaning when they are used alone

### Stemming

Next, I will extract the base form of words by removing affixes from them. This called stemming,there are numerous stemming algorithms,i'll use Snowball Stemmer

![](/images/stemming.png)


## Count Vectors

Machine learning algorithms cannot work with raw text directly. The text must be converted into numbers.
First, I will create a Bag of Words (BOW) model to extract features from text:

This algorithm is very similar to the one-hot encoding, but it has the advantage of identifying the frequency/counts of the words in the documents they appear.

- Step 1: Convert each document(email) into a sequence of words containing that document.

- Step 2: From the set of all the words in the corpus(collection of document), count how often the word occurs in the document.

![](/images/bag%20of%20words.png)


### Showing the  100 most used word in spam emails

![](/images/spamwords.png)

### Showing the  100 most used word in ham emails

![](/images/hamwords.png)

After that I searched the most used words for each category ,i will create a Bag of Word only with  that in order to reduce the number of variable that the model should  to handle

# Splitting data
Since the dataset is unbalanced i will split data using stratified sampling

# Training Models
I will use the following models:

|Model|Accuracy Test set|
|-----|--------|
|Naive Bayes| 95,81%|
|Logistic Regression| 97.82%|
|**Random Forest**| 98.42%| 
|Support Vector Classifier| 86.56%| 
|Voting Classifier |97.98%|

At this point before moving on I should choose the model to analyze,in
this case i will take into account only the accuracy on the test set to choose the model  (but I could also consider 'recall','precision' and others metrics),for this reason i will choose **Random Forest**

# Model's features

Showing the features of the Random Forest classifier

``` 
rf.get_params()

{'bootstrap': True,
 'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 100,
 'n_jobs': None,
 'oob_score': False,
 'random_state': 1,
 'verbose': 0,
 'warm_start': False}
 
 ```
 ![](/images/featureimportance.png)
 
 ![](/images/misclassification.png)
 
# Saving model with pickle and model Deployment
Pickle is the standard way of serializing objects in Python.
We can use the pickle operation to serialize our machine learning algorithms and save the serialized format to a file.

After saving the model I wrote the code to use it 

# Conclusion

I have obtained an accuracy of 98.42%  with a precision score  for spam emails of  98.23% and of 98.45% for ham emails,I can conclude saying that the model will work well with emails like in the dataset but since that modern emails have a more complex structure this model will not be useful in classifying those
