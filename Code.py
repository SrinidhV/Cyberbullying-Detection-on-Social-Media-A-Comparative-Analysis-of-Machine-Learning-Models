#===============================IMPORT PACKAGES===========================

import pandas as pd    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings

#===================== 1.READ A INPUT DATA ============================

data=pd.read_csv('Cyberbullying.csv')
print("*********************************************")
print()
print("Data Selection")
print()
print("********************************************")
print(data.head(10))
print()


#=====================  2.DATA PREPROCESSING ==========================


#=== CHECK MISSING VALUES ===

print("********************************************")
print()
print(" Handling missing values")
print()
print(data.isnull().sum())
print()

#=== DROP UNWANTED COLUMNS ===

data=data.drop(['annotation'], axis = 1)


#=================== 3.NLP TECHNIQUES ============================

#==== text cleaning ====

cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    return sentence


data["Summary_Clean"] = data["content"].apply(cleanup) 


print("********************************************")
print()
print("Before applying NLP techniques")
print()
print(data["content"].head(10))
print()
print("********************************************")
print()
print("After applying NLP techniques")
print()
print(data["Summary_Clean"].head(10))
print()

#=================== 4.SENTIMENT ANALYSIS ==========================


analyzer = SentimentIntensityAnalyzer()
data['compound'] = [analyzer.polarity_scores(x)['compound'] for x in data['Summary_Clean']]
data['neg'] = [analyzer.polarity_scores(x)['neg'] for x in data['Summary_Clean']]
data['neu'] = [analyzer.polarity_scores(x)['neu'] for x in data['Summary_Clean']]
data['pos'] = [analyzer.polarity_scores(x)['pos'] for x in data['Summary_Clean']]

#=== Labelling ===
data['comp_score'] = data['compound'].apply(lambda c: 0 if c >=0 else 1)


#================ 5.DATA SPLITTING =============================

X=data['Summary_Clean']
Y=data['comp_score']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=50)

#================= 6.FEATURE EXTRCTION ================================

vector = CountVectorizer(stop_words = 'english', lowercase = True)

#fitting the data
training_data = vector.fit_transform(X_train)

#tranform the test data
testing_data = vector.transform(X_test)   


#===================== 7.CLASSIFICATION ================================

#=== LINEAR REGRESSION ===

#initialize the model
clf = LinearRegression()

#fitting the model
clf.fit(training_data, Y_train)

#predict the model
predictions = clf.predict(testing_data)

#=== PERFROMANCE ANALYSIS ===

#calculate accuracy
Error_value=metrics.mean_absolute_error(Y_test,predictions)

Accuracy_Linear=100-Error_value

print("********************************************")
print()
print("--- PERFORMANCE ANALYSIS ---")
print()
print(" Accuracy for Linear regression :",Accuracy_Linear,'%')
print()


#=== Stochastic Gradient descent ===

#initialize the model
sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

#fitting the model
sgd.fit(training_data, Y_train)

#predict the model
predictions_sgd=sgd.predict(testing_data)

#=== PERFROMANCE ANALYSIS ===

#calculate accuracy
Error_value_sgd=metrics.mean_absolute_error(Y_test,predictions_sgd)

Accuracy_sgd=100-Error_value_sgd

print()
print("--- PERFORMANCE ANALYSIS ---")
print()
print(" Accuracy for Stochastic gradient descent :",Accuracy_sgd,'%')
print()
print("********************************************")
print()

#===================== 8.PREDICTION ================================

print("--- PREDICTION ---")
print()
print(predictions_sgd.shape)
z=int(input("Enter the id for finding Cyberbullying Case or Non Cyberbullying Case "))
if predictions_sgd[z] == 1:
    
    print('***********************************')
    print()
    print('--Cyberbullying case --')
    print()
    # print('***********************************')
    
else:
    
    # print('***********************************')
    print()  
    print('-- Non cyberbullying case --')
    print()
    print('***********************************')


#===================== 9.VISUALIZATION ================================

data['comp_score'] = data['comp_score'].replace([1 , 0] , ["Cyberbullying cases" , "Non Cyberbullying cases"])

plt.figure(figsize = (6,6))
counts = data['comp_score'].value_counts()
plt.pie(counts, labels = counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},autopct='%1.1f%%', pctdistance = 0.55, textprops = {'color': 'black', 'fontsize' : 9}, shadow = True,colors = sns.color_palette("Paired")[3:])
plt.text(x = -0.35, y = 0, s = 'Total tweets: {}'.format(data.shape[0]))
plt.title('Analysing cyberbullying cases', fontsize = 14);
plt.show() 




