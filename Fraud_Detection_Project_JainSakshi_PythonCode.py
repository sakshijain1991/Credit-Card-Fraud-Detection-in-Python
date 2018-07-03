
# coding: utf-8

# # Load Data Set and Import Packages

# In[1]:

import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


# In[2]:

get_ipython().magic('matplotlib inline')


# In[3]:

df = pd.read_csv("D:/Study/Ryder/creditcard.csv")


# # Data Exploration

# In[4]:

df.info()


# We have 28 anonymized features, amount of transaction, time of transaction and the class of transaction in the dataset.

# In[5]:

df.head()


# In[6]:

df.describe()


# In[7]:

df.isnull().sum()


# Result: There is no Missing Value in this dataset so no imputations are required.

# Target Column "Class". It has 2 values: 1- For Normal Transaction and 0- Fraud Transactions. Let us look at their numbers.

# In[8]:

count_classes = pd.DataFrame(pd.value_counts(df['Class'], sort = True).sort_index())
count_classes


# Result: Fraud transactions are only 492/(492+284315) = 0.1727% of total transactions.

# # Let's see how independent variables are affecting dependent variable.

# In[9]:

# Plot histogram of each parameter
df.hist(figsize = (20,20))
plt.show()


# Most of the features are clustered right around 0 with some fairly large outliers or no outliers. Also, very few fraud than normal transactions.

# In[10]:

# Correlation Matrix with the Heat Map

corrmat =  df.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax = 0.8, square = True)
plt.show()


# Lot of value really close to 0, so there is not strong relations between V parameters. But some are affecting Target variable i.e. "Class". So, V11 is strongly positive correlated where as V17 is strongly negative correlated with "Class". No Strong relation with Amount and Time. 

# Fraud and normal transaction vs. time: Let's see how time compares across fraudulent and normal transactions.

# In[11]:

print (df.Time[df.Class == 1].describe())
print ()
print ("Normal")
print (df.Time[df.Class == 0].describe())


# In[12]:

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 50

ax1.hist(df.Time[df.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(df.Time[df.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Number of Transactions')
plt.show()


# Result: Fraudulent are more uniformly distributed, while valid transactions have a cyclical distribution: Number of valid transactions is much smaller during the wee hours of the morning (between 1 to 5am). This could make it easier to detect a fraudulent transaction during at an 'off-peak' time.

# Fraud and normal transaction vs. Amount: Let's see how Amount compares across fraudulent and normal transactions.

# In[13]:

print ("Fraud")
print (df.Amount[df.Class == 1].describe())
print ()
print ("Normal")
print (df.Amount[df.Class == 0].describe())


# In[14]:

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 30

ax1.hist(df.Amount[df.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(df.Amount[df.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()


# Result: Most transactions are small amounts, less than 100. Fraudulent transactions have a maximum value far less than normal transactions, $2,125.87 vs $25,691.16.

# # Neural Network Using Keras

# Let us create the threshold of fraud transaction.

# In[15]:

import itertools
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical


# In[16]:

#df['Normal']=1-df['Class'], instead I am converting Class to categorical

df['Amount_max_fraud'] = 1
df.loc[df.Amount <= 2125.87, 'Amount_max_fraud'] = 0
df.head()


# # Setting Up Training an Test Data Sets

# Stratify Class column in train -test split to keep the same Fraud/Normal ratio in train and test data. I am not using any validation dataset here.

# In[17]:

train,test=train_test_split(df,test_size=0.2,random_state=0,stratify=df['Class'])# stratify the Class


# In[18]:

count_train = pd.value_counts(train['Class'], sort = True).sort_index()
count_test = pd.value_counts(test['Class'], sort = True).sort_index()
print (count_train) 
'\n'  
print(count_test)


# Drop target columns from model input datsets

# In[19]:

X_train = train.drop(['Class'], axis = 1)
X_test = test.drop(['Class'], axis = 1)


# Define target sets:

# In[20]:

Y_train = train.loc[:, ['Class']]
Y_test = test.loc[:, ['Class']]


# In[21]:

# Just sanity check
print(np.shape(X_train))
print(np.shape(Y_train))
print(np.shape(X_test))
print(np.shape(Y_test))


# In[22]:

#Now convert Y_train and Y_test to categorical values with 2 classes.

Y_train = to_categorical(Y_train, num_classes = 2)
Y_test = to_categorical(Y_test, num_classes = 2)


# # Feature Centering and scaling
# Centering and scaling of input datasets

# In[23]:

#Names all of the features in X_train.
features = X_train.columns.values

for feature in features:
    mean, std = df[feature].mean(), df[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std


# Now we start Keras model building

# In[24]:

# fix random seed for reproducibility
np.random.seed(2)


# Set up a 5 layer network with last layer being the output layer. First layer has input dimention as 31 (number of columns in X_train). Activation is relu except last layer is with softmax activation Each layer has dropout at 0.9 (90% of data used at each layer)

# In[25]:

model = Sequential()
model.add(Dense(64, input_dim=31, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(2, activation='softmax'))  # With 2 outputs


# Let us create X_test with this remaining 20% normal and fraud data

# Compile model using binary crossentropy loss and adam optimizer for loss. Collect accuracy in metric.

# In[26]:

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the compiled model on training data. I am using only 20 epochs to save time with batch_size of 2048.

# In[27]:

epoch = 20
batch_size = 2048
model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size)


# In[28]:

score, acc = model.evaluate(X_test, Y_test)
print('Test score:', score)
print('Test accuracy:', acc)


# We get 99.82% Accuracy!

# # Training and testing accuracy and loss vs epoch
# 
# Let us plot train and test accuracy and loss vs. epoch collcting the history by running the model again:

# In[29]:

history = model.fit(X_train, Y_train, batch_size = 2048, epochs = 20, 
         validation_data = (X_test, Y_test), verbose = 2)


# In[30]:

# Check the history keys
history.history.keys()


# In[31]:

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Testing loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Testing accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()


# # Confusion Matrix

# In[32]:

# Let us have a look at the confusion matrix for this 2 classes. I am using the below function to get the confusion matrix. 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[33]:

# Predict the values from the validation dataset
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2)) 
plt.show()


# Basically the model did not predict the Fraud transactions correctly. However, it did not predict any Normal transaction as Fraud. So, Let us try to do it other way

# # Logistic Regression

# In[34]:

data = pd.read_csv("D:/Study/Ryder/creditcard.csv")
data.head()


# In[35]:

count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()


# Clearly we can see that data is highly skewed with majority class of normal transactions.
# 
# Solution is UNDER-sampling, which deletes instances from the over-represented class

# # Set Predictor and Response Variable
# 
# 1. Normalising the amount column. The amount column is not in line with the anonymised features.

# In[36]:

from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
data.head()


# 2. Assigning X and Y. Resampling

# We will use UNDER-sampling. The way we will under sample the dataset will be by creating a 50/50 ratio. This will be done by randomly selecting "x" amount of sample from the majority class, being "x" the total number of records with the minority class.

# In[37]:

X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']


# In[38]:

# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# # Set Train and Test Data Set via Cross Validation

# In[39]:

from sklearn.cross_validation import train_test_split

# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

# Undersampled dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                   ,y_undersample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)
print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))


# In[40]:

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 


# In[41]:

def printing_Kfold_scores(x_train_data,y_train_data):
    fold = KFold(len(y_train_data),5,shuffle=False) 

    # Different C parameters
    c_param_range = [0.01,0.1,1,10,100]

    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        for iteration, indices in enumerate(fold,start=1):

            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C = c_param, penalty = 'l1')

            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())

            # Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration,': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.loc[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    
    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c


# In[42]:

best_c = printing_Kfold_scores(X_train_undersample,y_train_undersample)


# Confusion Matrix

# In[43]:

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# # Predictions on test set and plotting confusion matrix

# In[44]:

# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# Model give Recall Accuracy as 93.87. So let's apply the model we fitted and test it on the whole data.

# In[45]:

# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# Got Recall Accuracy as 92.5 which is quite decent!

#     # Plotting ROC curve and Precision-Recall curve.

# In[46]:

# ROC CURVE
lr = LogisticRegression(C = best_c, penalty = 'l1')
y_pred_undersample_score = lr.fit(X_train_undersample,y_train_undersample.values.ravel()).decision_function(X_test_undersample.values)

fpr, tpr, thresholds = roc_curve(y_test_undersample.values.ravel(),y_pred_undersample_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # One Class SVM Linear

# In[47]:

from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report


# In[48]:

W_Data=pd.read_csv("D:/Study/Ryder/creditcard.csv")
W_Data.dropna(thresh=284807)
Data=W_Data


# In[49]:

Positives = W_Data[Data['Class']==1]
Negatives = W_Data[Data['Class']==0]


# In[50]:

print((len(Positives)/len(Data))*100,"%")


# In[51]:

Train_Data=Data[1:50000]
Target=Train_Data['Class']
Train_Data.drop('Class',axis=1,inplace=True)


# In[52]:

x_train,x_test,y_train,y_test=train_test_split(Train_Data,Target,test_size=0.5,random_state=0)


# In[53]:

from sklearn.covariance import EllipticEnvelope 
#An Object for detecting outliers in a Gaussian distributed dataset


# In[54]:

#Linear Kernel
clf_AD_L = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
clf_AD_L.fit(Negatives)


# In[55]:

train_AD_L=clf_AD_L.predict(Negatives)
test_AD_L=clf_AD_L.predict(Positives)


# In[56]:

def Train_Accuracy(Mat):
   
   Sum=0
   for i in Mat:
    
        if(i==1):
        
           Sum+=1.0
            
   return(Sum/len(Mat)*100)

def Test_Accuracy(Mat):
   
   Sum=0
   for i in Mat:
    
        if(i==-1):
        
           Sum+=1.0
            
   return(Sum/len(Mat)*100)


# In[57]:

print("Training: One Class SVM (Linear) : ",(Train_Accuracy(train_AD_L)),"%")
print("Test: One Class SVM (Linear) : ",(Test_Accuracy(test_AD_L)),"%")


# # Isolation Forest

# In[58]:

from sklearn.ensemble import IsolationForest


# In[59]:

IFA=IsolationForest()
IFA.fit(Negatives)


# In[60]:

train_IFA=IFA.predict(Negatives)
test_IFA=IFA.predict(Positives)


# In[61]:

print("Training: Isolation Forest: ",(Train_Accuracy(train_IFA)),"%")
print("Test: Isolation Forest: ",(Test_Accuracy(test_IFA)),"%")


# Isolation Forest has worked way better than one class SVM. Thus, considered as best anomaly detection model.
