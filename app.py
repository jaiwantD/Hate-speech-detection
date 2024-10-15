from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import nltk
import re
import string
from nltk.corpus import stopwords

app= Flask(__name__)

stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

# Load the dataset
data = pd.read_csv(r"C:\Users\kishore giri\Downloads\HateSpeechData.csv")

# Preview the data
# print(data.head())

# Map class values to readable labels
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]


# Cleaning the unwanted !@#$%^&*()
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



#model


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



# routing
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = clean(text)
        vect = cv.transform([cleaned_text]).toarray()
        prediction = model.predict(vect)
        return render_template('index.html', prediction=prediction[0])


if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pandas as pd
# import nltk
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.tree import DecisionTreeClassifier
# import re
# import string
# from nltk.corpus import stopwords

# # Initialize Flask app
# app = Flask(__name__)

# # Load stopwords and stemmer
# stopword = set(stopwords.words('english'))
# stemmer = nltk.SnowballStemmer("english")

# # Load the dataset and preprocess
# data = pd.read_csv(r"C:\Users\kishore giri\Downloads\HateSpeechData.csv")
# data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
# data = data[["tweet", "labels"]]

# # Define cleaning function
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

# # Apply cleaning to data
# data["tweet"] = data["tweet"].apply(clean)

# # Convert text data into numerical form
# cv = CountVectorizer()
# X = cv.fit_transform(data["tweet"]).toarray()
# y = np.array(data["labels"])

# # Model building and training
# model = DecisionTreeClassifier()
# model.fit(X, y)

# # Home route
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Real-time prediction route for AJAX
# @app.route('/ajax_predict', methods=['POST'])
# def ajax_predict():
#     text = request.get_json().get("text")
#     cleaned_text = clean(text)
#     vect = cv.transform([cleaned_text]).toarray()
#     prediction = model.predict(vect)
#     return jsonify({'prediction': prediction[0]})

# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, render_template, request
# import numpy as np
# import pandas as pd
# import nltk
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.tree import DecisionTreeClassifier
# import re
# import string
# from nltk.corpus import stopwords

# # Initialize Flask app
# app = Flask(__name__)

# # Load stopwords and stemmer
# nltk.download('stopwords')
# stopword = set(stopwords.words('english'))
# stemmer = nltk.SnowballStemmer("english")

# # Load the dataset and preprocess
# data = pd.read_csv(r"C:\Users\kishore giri\Downloads\HateSpeechData.csv")
# data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
# data = data[["tweet", "labels"]]

# # Define cleaning function
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

# # Apply cleaning to data
# data["tweet"] = data["tweet"].apply(clean)

# # Convert text data into numerical form
# cv = CountVectorizer()
# X = cv.fit_transform(data["tweet"]).toarray()
# y = np.array(data["labels"])

# # Model building and training
# model = DecisionTreeClassifier()
# model.fit(X, y)


