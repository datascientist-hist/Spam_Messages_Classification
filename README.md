# Email Spam Classification

<img src="/images/email-spam.png" width="200" >

## Goal

The aim of this project is to classify emails into spam and non-spam email 

To do this I will use the frequencies method ,that will count the how many different words there are in a sentence after that I will choose a number of relevant words.Below I will expalin better the method

## Libraries used

```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly_express as px
import plotly.figure_factory as ff
from wordcloud import WordCloud
import nltk
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

```

## Loading Dataset

Dataset has been downloaded from kaggle.You can see more info clicking **[here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)**
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

## Splitting data
Since the dataset is unbalanced i will split data using stratified sampling

# Training Models

I will use the following models to classify the target variable:

|Model|Accuracy Baseline|
|-----|--------|
|Naive Bayes| 87.83%|
|Logistic Regression| 97.02%|
|**Random Forest**| 98.22%| 
|Support Vector Classifier| 86.59%| 
|Voting Classifier |97.45%|
