'''This is a Machine learning app Created by the use of Streamlit app'''

# Importing the Libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")
st.sidebar.header('User Input Parameters')

# Function for the Slider
def user_input_features():
    sepal_lenght = st.sidebar.slider('Sepal Length',4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {'sepal_lenght': sepal_lenght,
            'sepal_width' : sepal_width,
            'petal_length': petal_length,
            'petal_width' : petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

dframe = user_input_features()

st.subheader('**User Input Parameters**')
st.write(dframe)

# Load the Dataset
dt = pd.read_csv(r"iris.data", header=None)
col = ['sepal length','sepal width','petal length','petal width','class']
dt.columns = col
st.write("### **Iris DataSet** :", dt.head(10))

# Extracting the feature matrics
X = dt.iloc[:,:-1].values
y = dt.iloc[:,-1].values

# st.write('**data** :', X,'\t','**targets** :', y)
# print(X)
# print(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# st.write(X_train)


# Performing the Standardization of the input variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training the  Random Forest model on the Training set
clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
clf.fit(X_train,y_train)

pred = clf.predict(sc.transform(dframe.values))
pred_proba = clf.predict_proba(sc.transform(dframe.values))

# print(pred)

st.subheader('**Class labels and their corresponding index number**')
dict1 = {'Iris-setosa': 0,
         'Iris-versicolor': 1,
         'Iris-virginica': 2}
classes = pd.DataFrame(dict1, index=[0])
st.write(classes)


st.subheader('**Prediction**')
st.write(classes[pred])
 

st.subheader('**Prediction Probability**')
st.write(pred_proba)