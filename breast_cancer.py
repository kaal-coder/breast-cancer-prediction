# Importing necessary libraries
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Loading the breast cancer dataset
breast_cancer_dataset = pd.read_csv('/content/breast cancer.csv')

# Loading the data to a data frame
data_frame = breast_cancer_dataset

# Adding the 'diagnosis' column to the data frame
data_frame['diagnosis'] = breast_cancer_dataset.diagnosis

# Checking the structure of the data
data_frame.shape
data_frame.info()
data_frame.isnull().sum()
data_frame.describe()
data_frame['diagnosis'].value_counts()

# Grouping the data by diagnosis
data_frame.groupby('diagnosis').mean()

# Separating the features and target
X = data_frame.drop(columns=['id', 'diagnosis'])
Y = data_frame['diagnosis']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Training a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluating the model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# Building a predictive system
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)

# Applying K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Training a SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, Y_train)

# Evaluating the SVM model
Y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

# Training a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)

# Evaluating the Decision Tree classifier
Y_pred = clf.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)

# Plotting the accuracy of different methods
methods = ['SVM', 'K clustering', 'Logistic regression', 'Decision Trees']
accuracies = [0.956, 0.558, 0.929,0.912 ]
plt.bar(methods, accuracies, color='blue')
plt.ylim([0.0, 1.0])
plt.xlabel('Methods')
plt.ylabel('Accuracy')
plt.title('Accuracy of methods')
plt.show()

# Saving the trained model for deployment
import pickle
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))   #wb-write binary

# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb')) #rb-read binary

# Predicting using the saved model
input_data = (20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,
              0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)