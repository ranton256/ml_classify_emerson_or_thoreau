# Supervised learning for Text Classification: Emerson or Thoreau

## NLP - Text Classification Model
This is a walkthrough of different ML classification models to classify short passages of text between two authors.

The dataset is constructed from two public domain texts.

The notebook code shows how to perform typical NLP text preprocessing tasks and exploratory data analysis.

This covers and compares multiple classification model algorithms from logistic regression with TF-IDF, random forests, SVM on pretrained DistilBERT weights, and finetuning a pretrained DistilBERT model.

## The Dataset

Our two datasets are constructed from two related works of 19th century American transcendentalism. These are both public domain.

- [Essays by Ralph Waldo Emerson by Ralph Waldo Emerson](https://www.google.com/url?q=https%3A%2F%2Fwww.gutenberg.org%2Febooks%2F16643)
- [Walden, and On The Duty Of Civil Disobedience by Henry David Thoreau](https://www.google.com/url?q=https%3A%2F%2Fwww.gutenberg.org%2Febooks%2F205)

These two authors had different writing styles but shared more than their philosophical interests—they were neighbors in Concord, Massachusetts.

These two works are also similar in length when formatted as plain text.

We will use spaCy to segment each work into sections of roughly 3 to 5 sentences each, then build a datafrom of the text including a label of 'emerson' or 'thoreau', then shuffle and split that into train and test sets for training some machine learning models to classify them by predicting which author they are from and compare the results.

We will also preprocess text to remove stopwords, and perform lemmatization.

# Setup 

To run the code you need to install the dependencies first.
Setup a fresh virtual env, for example:

`python -mvenv .venv`

Activate the enviornment:

`source .venv/bin/activate`

Then install the requirements.

`pip install -r requirements.txt`

Start Jupyter if running locally:

`juypyter-lab supervised_ML_identify_author.ipynb`

If you want to setup a new kernel:
`ipython kernel install --name "local-venv" --user`



