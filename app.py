import pandas as pd
import numpy as np
import flask as fl

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV

#read in file data
df = pd.read_table("SMSSpamCollection" , sep='\t' ,header =None)
df = df.reindex(np.random.permutation(df.index))
df.reset_index(inplace=True, drop=True)
df.head()

df.columns=["label", "sms_message"] #label the columns as sms_message
df['label']=df['label'].replace('ham', 0) #replace every ham in column 'label' with the binary number 0
df['label']=df['label'].replace('spam', 1) #replace every spam in column 'label' with the binary number 1

# the split ratio is 25% by default
# Here df['sms_message'] is X and df['label'] is y for both test and training data
X_train_sms, X_test_sms, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=42)
#test data distribution worked correctly
print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(y_train.shape[0]))
print('Number of rows in the test set: {}'.format(y_test.shape[0]))

# Instantiate the CountVectorizer method
count_vector = CountVectorizer(ngram_range=(1, 1), lowercase = True , stop_words =  'english')

# Fit the training data and then return the matrix
X_train = count_vector.fit_transform(X_train_sms) 


# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
X_test = count_vector.transform(X_test_sms)

#get a list of the words, or the column labels, that correspond to each word
X_train_feature_list = count_vector.get_feature_names_out()

#convert the training count vector to an array and then turn that array into a matrix
doc_array =  X_train.toarray()
frequency_matrix_X_train = pd.DataFrame((doc_array),columns = X_train_feature_list)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train , y_train)

predictions = naive_bayes.predict(X_test)
print('\nWord Count results using Naive Bayes:')
print('Accuracy score: ', format(accuracy_score(predictions,y_test)))
print('Precision score: ', format(precision_score(predictions,y_test)))
print('Recall score: ', format(recall_score(predictions,y_test)))
print('F1 score: ', format(f1_score(predictions,y_test)) + "\n \n") 


#testing TF-IDF vs just word count
tfIdfVectorizer = TfidfVectorizer(use_idf=True, lowercase = True , stop_words =  'english')
tfIdf_X_train = tfIdfVectorizer.fit_transform(X_train_sms)
tfIdf_X_test = tfIdfVectorizer.transform(X_test_sms)

#get a list of the words, or the column labels, that correspond to each word
tfIdf_X_train_feature_list = tfIdfVectorizer.get_feature_names_out()

tfIdf_doc_array =  tfIdf_X_train.toarray()
frequency_matrix_tfIdf_X_train = pd.DataFrame((tfIdf_doc_array),columns = tfIdf_X_train_feature_list)

tfIdf_naive_bayes = MultinomialNB()
tfIdf_naive_bayes.fit(tfIdf_X_train , y_train)

tfIdf_predictions = tfIdf_naive_bayes.predict(tfIdf_X_test)

print('TF-IDF results using Naive Bayes:')
print('Accuracy score: ', format(accuracy_score(tfIdf_predictions,y_test)))
print('Precision score: ', format(precision_score(tfIdf_predictions,y_test)))
print('Recall score: ', format(recall_score(tfIdf_predictions,y_test)))
print('F1 score: ', format(f1_score(tfIdf_predictions,y_test))) 


#random forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_y_predictions = rf.predict(X_test)

print('\nWord Count results using Random Forest Classifier:')
print('Accuracy score: ', format(accuracy_score(rf_y_predictions,y_test)))
print('Precision score: ', format(precision_score(rf_y_predictions,y_test)))
print('Recall score: ', format(recall_score(rf_y_predictions,y_test)))
print('F1 score: ', format(f1_score(rf_y_predictions,y_test))) 

#random forest classifier using halving random search to find optimal parameters for forest
rng = np.random.RandomState(0)

param_dist = {
    "max_depth": [4, None],
    #"min_samples_split": np.random.randint(2, 11),
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
}

rsh = HalvingRandomSearchCV(estimator=rf, param_distributions=param_dist, factor=2, random_state=rng)
rsh.fit(X_train, y_train)
rsh_y_predictions = rsh.predict(X_test)

print('\nWord Count results using Random Forest Classifier with HalvingRandomSearch:')
print('Accuracy score: ', format(accuracy_score(rsh_y_predictions,y_test)))
print('Precision score: ', format(precision_score(rsh_y_predictions,y_test)))
print('Recall score: ', format(recall_score(rsh_y_predictions,y_test)))
print('F1 score: ', format(f1_score(rsh_y_predictions,y_test))) 

#docker stuff
app = fl.Flask(__name__)

@app.route('/')
def hello_world():
    rfstr = '\nWord Count results using Random Forest Classifier:' + '\n' + 'Accuracy score: ' + format(accuracy_score(rf_y_predictions,y_test)) + '\n' + 'Precision score: ' + format(precision_score(rf_y_predictions,y_test)) + "\n" + 'Recall score: ' + format(recall_score(rf_y_predictions,y_test)) + '\n' + 'F1 score: ' + format(f1_score(rf_y_predictions,y_test))
    return 'Bag of Words\nAccuracy score: '+ format(accuracy_score(predictions,y_test)) + '\n' + 'Precision score: ' + format(precision_score(predictions,y_test)) + "\n" + 'Recall score: ' + format(recall_score(predictions,y_test)) + '\n' + 'F1 score: ' + format(f1_score(predictions,y_test)) + '\n' + rfstr