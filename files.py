import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import textstat
from scipy.stats import pearsonr
from wordcloud import WordCloud
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, cohen_kappa_score
from scipy.sparse import hstack
import numpy as np
import scipy.sparse


# Calculate the Token-Type Ratio (TTR) to measure lexical diversity


def calculate_ttr(text):
    """
    Calculate the Token-Type Ratio (TTR) of a text.
    TTR is the ratio of unique tokens to the total number of tokens in the text.
    
    :param text: The input text to analyze
    :return: The TTR as a float
    """
    tokens = word_tokenize(text)
    types = set(tokens)
    return len(types) / len(tokens) if tokens else 0

# Calculate average sentence length as a measure of syntactic complexity
def calculate_avg_sentence_length(text):
    """
    Calculate the average sentence length in a text.
    This is a proxy for syntactic complexity.
    
    :param text: The input text to analyze
    :return: The average sentence length as a float
    """
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    return len(tokens) / len(sentences) if sentences else 0

# Calculate Guiraud's index as another measure of lexical richness
def calculate_guiraud_index(text):
    """
    Calculate Guiraud's index for the text.
    It's another measure of lexical richness, defined as the ratio of unique tokens to the square root of total tokens.
    
    :param text: The input text to analyze
    :return: Guiraud's index as a float
    """
    tokens = word_tokenize(text)
    types = set(tokens)
    return len(types) / (len(tokens)**0.5) if tokens else 0

# Calculate sentiment polarity score to analyze text sentiment
def calculate_sentiment(text):
    """
    Calculate the sentiment polarity of a text using TextBlob.
    The polarity score ranges from -1 (very negative) to 1 (very positive).
    
    :param text: The input text to analyze
    :return: Sentiment polarity score as a float
    """
    return TextBlob(text).sentiment.polarity

# Additional functions seem well-structured but ensure all necessary libraries are imported and initialized.

# Clean essay text by removing special characters, digits, and converting to lowercase
def clean_essay_new(essay):
    """
    Clean essay text by removing special characters, numbers, and converting to lowercase.
    
    :param essay: The essay text to clean
    :return: The cleaned essay text
    """
    essay = re.sub(r'@\w+', '', essay)  # Removing special markers
    essay = essay.lower()  # Converting to lowercase
    essay = re.sub(r'[^\w\s]', '', essay)  # Removing punctuation
    essay = re.sub(r'\d+', '', essay)  # Removing digits
    return essay



def syntactic_complexity(text):
    sentences = sent_tokenize(text)
    if not sentences: return 0
    return sum(len(sentence.split()) for sentence in sentences) / len(sentences)

def plot_correlation_with_target(df, target_column):
    """
    Plots a heatmap to visualize the correlation between various text metrics and a target column.
    
    This function selects specific columns of interest, calculates the correlation matrix for these columns,
    and then plots the matrix as a heatmap. It's useful for understanding the relationships between text features
    and the target variable (e.g., essay scores).
    
    :param df: Pandas DataFrame containing the data.
    :param target_column: The name of the target column against which correlations are to be calculated.
    """
    # Select columns for correlation calculation
    columns_of_interest = ['ttr', 'avg_sentence_length', 'guiraud_index', 'sentiment_polarity',
                           'flesch_reading_ease', 'gunning_fog', 'lexical_diversity', target_column]
    
    # Calculate correlation matrix for selected columns
    correlation_matrix = df[columns_of_interest].corr()
    
    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    
    # Title for the heatmap
    plt.title(f'Correlation with {target_column}')
    
    # Display the heatmap
    plt.show()
    
# Clean essay text by removing special characters, digits, and converting to lowercase
def clean_essay_new(essay):
    """
    Clean essay text by removing special characters, numbers, and converting to lowercase.
    
    :param essay: The essay text to clean
    :return: The cleaned essay text
    """
    essay = re.sub(r'@\w+', '', essay)  # Removing special markers
    essay = essay.lower()  # Converting to lowercase
    essay = re.sub(r'[^\w\s]', '', essay)  # Removing punctuation
    essay = re.sub(r'\d+', '', essay)  # Removing digits
    return essay


def qwk(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK), a measure of agreement between two ratings.
    This metric is particularly useful for evaluating the agreement between human and machine ratings.
    """
    return cohen_kappa_score(np.round(y_true), np.round(y_pred), weights='quadratic')
