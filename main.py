import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

labelencoder =LabelEncoder()
def data_visualiser(df):
    st.subheader("Data Visualisation: ")
    if df is not None:
        df= df.drop(columns=['Name'])
        st.dataframe(df)

def classification(model_chooser):
    if model_chooser=='SVC':
        C_value = st.sidebar.slider('C-value',0.01,10.00)
        gamma = st.sidebar.slider('Gamma',0.01,10.00)
        clf = SVC(C=C_value,kernel='rbf',random_state=0)

    elif model_chooser =='Random Forest':
        max_depth = st.sidebar.slider('Max Depth',1,15)
        no_estimators = st.sidebar.slider("No of estimators",1,100)
        clf = RandomForestClassifier(n_estimators=no_estimators,max_depth=max_depth,criterion='entropy',random_state=0)


    elif model_chooser =='KNN':
        K_value = st.sidebar.slider('K-value',1,20)
        clf = KNeighborsClassifier(n_neighbors=K_value,metric='minkowski',p=2)
    return clf


def survival():
    Pclass = st.sidebar.slider('Pclass', 1, 3)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 0, 80)
    n_sibling_spouse = st.sidebar.slider('Siblings/Spouses', 0, 5)
    n_parents_children = st.sidebar.slider('Parents/Children', 0, 3)
    fare = st.sidebar.slider('Fare', 0, 200)
    if sex == 'Male':
        sex = 1
    else:
        sex = 0

    features = [[Pclass, sex, age, n_sibling_spouse, n_parents_children, fare]]
    return features




#Title of web app
st.write("""
# Titanic Survival Predictor 
Use Machine Learning to predict survival based on the Titanic Dataset!
""")

#Web Image
img = Image.open("titanic_bg.jpg")
st.image(img,use_column_width=True)
#Main app
df=pd.read_csv("titanic.csv")
data_visualiser(df)
user_visuals = st.sidebar.selectbox('Select Visualisation',('Shape','Data Description','Number of Survivors','Survivors by Class,Gender and Age'))

if user_visuals == 'Shape':
    st.subheader('Shape of Data: ')
    st.write(df.shape)
elif user_visuals == 'Data Description':
    st.subheader('Description: ')
    st.write(df.describe())
elif user_visuals == 'Number of Survivors':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('Number of Survivors: ')
    st.write(df.iloc[:,0].value_counts())
    st.subheader("Survivor Visualisation ")
    sns.countplot(df['Survived'])
    st.pyplot()
elif user_visuals =='Survivors by Class,Gender and Age':
    st.subheader("Survivors by Class,Gender and Age")
    fig, axs= plt.subplots(3,1)
    sns.barplot(x='Pclass',y='Survived',data=df,ax=axs[2])
    sns.barplot(x='Sex', y='Survived', data=df, ax=axs[1])
    age = pd.cut(df['Age'],[0,18,30,45,60,80])
    b=sns.barplot(x=age,y='Survived',data=df,ax=axs[0])
    b.tick_params(labelsize=5)
    fig.tight_layout(pad=1)
    st.pyplot(fig)

model_chooser = st.sidebar.selectbox('Select Model',('SVC','Random Forest','KNN'))
clf = classification(model_chooser)


df= df.drop(columns=['Name'])
#encode the sex column
df.iloc[:,2]=labelencoder.fit_transform(df.iloc[:,2].values)
st.dataframe(df.head())
#split the data into independent x and dependent y variables
x=df.iloc[:,1:7].values
y=df.iloc[:,0].values
#split the data into 80% training and 20% testing
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
st.write(f"Classifier={model_chooser}")
st.write(f"Accuracy={accuracy}")

survive = survival()
prediction = clf.predict(survive)

if prediction==0:
    st.write("You did not survive")
else:
    st.write("You survived!")



