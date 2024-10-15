# # import pandas as panda
# # from nltk.tokenize import word_tokenize
# # from nltk.corpus import stopwords
# # from nltk.stem.porter import *
# # import string
# # import nltk
# # from sklearn.feature_extraction.text import CountVectorizer
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics import confusion_matrix
# # import seaborn
# # from textstat.textstat import *
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import f1_score
# # from sklearn.feature_selection import SelectFromModel
# # from sklearn.metrics import classification_report
# # from sklearn.metrics import accuracy_score
# # from sklearn.svm import LinearSVC
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.naive_bayes import GaussianNB
# # import numpy as np
# # from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
# # import warnings
# # warnings.simplefilter(action='ignore', category=FutureWarning)


# # dataset = panda.read_csv(r'C:\Users\kishore giri\Downloads\HateSpeechData.csv')
# # dataset['text length'] = dataset['tweet'].apply(len)
# # print(dataset.head())

# #Importing the packages
# # Importing the packages
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# import nltk
# import re
# import string
# from nltk.corpus import stopwords

# # Download NLTK stopwords
# nltk.download('stopwords')

# # Set stopwords and stemmer
# stopword = set(stopwords.words('english'))
# stemmer = nltk.SnowballStemmer("english")

# # Load the dataset
# data = pd.read_csv(r"C:\Users\kishore giri\Desktop\VSsetup\labeled_data.csv")

# # Preview the data
# print(data.head())

# # Map class values to readable labels
# data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
# data = data[["tweet", "labels"]]
# print(data.head())

# # Define a function to clean the text
# def clean(text):
#     text = str(text).lower()
#     text = re.sub('[.?]', '', text)
#     text = re.sub('https?://\S+|www.\S+', '', text)
#     text = re.sub('<.*?>+', '', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub('\n', '', text)
#     text = re.sub('\w*\d\w*', '', text)
#     text = [word for word in text.split() if word not in stopword]
#     text = " ".join(text)
#     text = [stemmer.stem(word) for word in text.split()]
#     return " ".join(text)

# # Apply the clean function to the tweets
# data["tweet"] = data["tweet"].apply(clean)

# # Convert text data into numerical form
# x = np.array(data["tweet"])
# y = np.array(data["labels"])

# cv = CountVectorizer()
# X = cv.fit_transform(x)

# # Splitting the Data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # Model building
# model = DecisionTreeClassifier()

# # Training the model
# model.fit(X_train, y_train)

# # Testing the model
# y_pred = model.predict(X_test)

# # Accuracy Score of our model
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, y_pred))

# # Predicting the outcome for a sample input
# inp = input("Enter the sentence :")
# inp = cv.transform([inp]).toarray()
# print(model.predict(inp))
# '''
# inp = "You are a great person"
# inp = cv.transform([inp]).toarray()
# print(model.predict(inp))
# inp = "You are a shit thing in world"
# inp = cv.transform([inp]).toarray()
# print(model.predict(inp))
# inp =" up, and we’d u on board!And guys, this is an immediate process as we’re starting the project very soon. You’ll be jumpin on’t be much time for learning on the job. We’re looking for people who already have hands-on experience and are ready to perform right from the start."
# inp = cv.transform([inp]).toarray()
# print(model.predict(inp))'''
# import pandas as panda
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem.porter import *
# import string
# import nltk
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import confusion_matrix
# import seaborn
# from textstat.textstat import *
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
# from sklearn.feature_selection import SelectFromModel
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# import numpy as np
# from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)


# dataset = panda.read_csv(r'C:\Users\kishore giri\Downloads\HateSpeechData.csv')
# dataset['text length'] = dataset['tweet'].apply(len)
# print(dataset.head())

#Importing the packages
# Importing the packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import nltk
import re
import string
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Set stopwords and stemmer
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

# Load the dataset
data = pd.read_csv(r"C:\Users\kishore giri\Downloads\HateSpeechData.csv")

# Preview the data
print(data.head())

# Map class values to readable labels
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]
print(data.head())

# Define a function to clean the text
def clean(text):
    text = str(text).lower()
    text = re.sub('[.?]', '', text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

# Apply the clean function to the tweets
data["tweet"] = data["tweet"].apply(clean)

# Convert text data into numerical form
x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x)

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Model building
model = DecisionTreeClassifier()

# Training the model
model.fit(X_train, y_train)

# Testing the model
y_pred = model.predict(X_test)

# Accuracy Score of our model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Predicting the outcome for a sample input
inp = "You are too bad and I don't like your attitude"
inp = cv.transform([inp]).toarray()
print(model.predict(inp))
inp = "You are a great person"
inp = cv.transform([inp]).toarray()
print(model.predict(inp))
inp = "You are a shit thing in world"
inp = cv.transform([inp]).toarray()
print(model.predict(inp))
inp = "You are a great to admire "
inp = cv.transform([inp]).toarray()
print(model.predict(inp))