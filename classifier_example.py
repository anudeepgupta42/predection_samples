import os
os.chdir(r"D:\EAISMSD\Banking and Finance\BankCustomer_Churn")

"""
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../InputData"]).decode("utf8"))
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df = pd.read_csv("Churn_Modelling.csv")
df.head() 

### Step 01: Data Exploration ###

## 1.1 Univariate Analsys ##
# describe() gives us the count, mean, std, min, q25%,50%, 75% of non-catogerical cols
df.describe() 

# gives the notnull values along with data t
df.info()

""" 
from head(), describe(),info() we can see that the variables Tenure, NumOfProducts, HasCrCard
IsActiveMember, Exited(traget variable) are catogerical in nature but are defined as int because of their values.
lets see the diff levels and count of each unique value in the above mentioned cols
"""

# unique() gives the unique values present variable
df.Geography.unique() #France, spain,Germany
df.Gender.unique() # Male, Female

#df['Geography'].unique() # same as df.<col>.unique()
intcols =['Tenure','NumOfProducts','HasCrCard','IsActiveMember','Exited' ]
print('col_name : Unique values')
for col in intcols:
    print(col, ':', df[col].unique())

"""
col_name : Unique values
Tenure : [2 1 8 ... 5 9 0]
NumOfProducts : [1 3 2 4]
HasCrCard : ['1' 'na' '0']
IsActiveMember : [1 0]
Exited : [1 0]
"""

# shape Gives the no. of unique values
df.Geography.unique().shape #(3,)
df.Gender.unique().shape #(2,)
print('col_name : no.of Unique values')
for col in intcols:
    print(col, '   :', df[col].unique().shape)
    
"""
col_name : no.of Unique values
Tenure    : (11,)
NumOfProducts    : (4,)
HasCrCard    : (3,)
IsActiveMember    : (2,)
Exited    : (2,)
"""
    
# value_counts() gives the count of each unique value in descending order
df['Geography'].value_counts()
"""
France     5014
Germany    2509
Spain      2477
Name: Geography, dtype: int64
"""
for col in intcols:
    print(col, '   :', df[col].value_counts())
    

df = df.iloc[:,3:14]
df.dtypes
##1.2 Treating the missing values##
""" in our data missing values are represented as na, we need to replace this na with NAN"""
df=df.replace('na', np.NaN) # replaces 'na' with NAN 

df.isnull().sum() # gives the no.of missing values in each variable/col
df.isnull().any()
df[df.isnull().any(axis=1)] # prints the rows with NaN values
# we can handle missing values either with fillna() or imputer from sklearn
#df.fillna()

## get the list of col which has missing values
missingCol=[]
for col in df.columns:
  if df[col].isnull().any():
      missingCol.append(col)

###converting catogerical variables to object type and continious to float
colToFloat=['CreditScore','Age','Balance','EstimatedSalary']
colToObject=['Tenure','NumOfProducts','HasCrCard','IsActiveMember','Exited']
for col in colToFloat:
    df[col] = df[col].astype('float64',copy=False)

for col in colToObject:
    df[col] = df[col].astype('object',copy=False)   
df.dtypes
## below imputation can be done using impuer as well
"""
from sklearn.preprocessing import Imputer
df[col] = Imputer(strategy='median/mean').fit_transform(df[col].values.reshape(-1, 1))
"""
for col in missingCol:
    if (df[col].dtypes == 'object'):
        print(col,' categorical', df[col].mode())
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        print(col,' continious', df[col].mean())
        df[col].fillna(df[col].mean(), inplace=True)
# when we want to fill na with mean of a particular group but not the entire col use below
#df[col1] = df.groupby(col2).transform(lambda x: x.fillna(x.mean()))
df.isnull().sum().sum() # this time it should return 0
df.shape
##1.3 Bivariate Analsys 
##Corr() gives corelation btw continiuos variables
# default method is perason and is widely used
corr_matrix = df.corr(method='pearson')
import seaborn as sns
ax=sns.heatmap(corr_matrix,linewidth=0.5)
##chisquare test has tobe performed to get the dependecy btw catogerical variables
##Ztest has tobe performed to get the dependecy btw catogericaland continious variables

### Step 2:  Outlier detection and handling ###
from scipy import stats
contCol=['CreditScore','Age','Balance','EstimatedSalary']
df.shape # return 10000 rows
df = df[(np.abs(stats.zscore(df[contCol])) < 3).all(axis=1)]
df.shape # returns 9859 rowns i.e outlier data is removed

## 1.2 pre-processing cont..... ##
# converting catogerical to numerical or continiuos values

# boolean to binary
bool_cols=[] # list all the columns which has boolean values
for col in bool_cols:
        df[col] = (df[col] == 'Yes/True').astype(int)

df.dtypes
## label encoding catogerical variables using cat.codes on catogery type of variables
# we can convert object to catogery type and use cat.codes to label encode
df_geo_catcode =df['Geography'].astype('category',copy=False)
df_geo_catcode.dtypes # prints catogery as dtype
df_geo_catcode.head() 
df_geo_catcode = df_geo_catcode.cat.codes # converts the datatype to int
df_geo_catcode.head() 
df_geo_catcode =df_geo_catcode.astype('object',copy=False) # converting back to object

# created a new df_geo_catcode to demonstarte the usage of 

# Alternatively we can use labelencoder from sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
#list(df.select_dtypes(include = ['object']).columns) this will list all teh object i.e catogerical col
cat_cols = ['Gender']  #list the catogerical variables to label encode them 
#  excluded geography as it is already encoded in above step
# 'Tenure','NumOfProducts','HasCrCard','IsActiveMember', are already in in encoded state
# since gender has only two factors we can need not process it further (it is equuivalent to binary)
for col in cat_cols:
    le = LabelEncoder()
    le = le.fit(df[col])
    df[col] = le.fit_transform(df[col])
df[cat_cols].head(10)
df.shape
# onehot encoding is used in the case when catogerical level/factors are more than 2

## using LabelBinarizer for onehot encoding
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb = lb.fit(df["Geography"])

df_LB = lb.transform(df["Geography"]) # returns a matrix
df_geo_labelBinariser = pd.DataFrame(df_LB, columns=['France','Germany', 'Spain']) # converting the aove matrix to a dataframe
df['Geography'].head()
# demo of using getdummies from pandas
df_Geo_getDummies = pd.get_dummies(df['Geography'])

# we can use onehotencoder from sk learn or getdummies from pandas

# more time to be spent to know the usage of OneHotEncoder on data frames

"""
df_temp=df
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
df_temp['Geography'] = le.fit_transform(df_temp['Geography'])
countryhotencoder = OneHotEncoder(categorical_features = [1]) # 1 is the country column
df_temp = countryhotencoder.fit_transform(df_temp).toarray()
"""
#featurehasing to be used when there are high no. of factors/level for a catogerical variable
from sklearn.feature_extraction import FeatureHasher
fh_1col = FeatureHasher(n_features=1, input_type='string')

fh_Tenure_1col = fh_1col.fit(df['Geography'])

df_fh_geo_1col = fh_Tenure_1col.transform(df['Geography'])
df_fh_geo_1col = pd.DataFrame(df_fh_geo_1col.toarray(), columns=['Geo_onehash'])


##### Feature Engineering and transformation
"""
once preprocessed perform some data engineeringif required and do sample(strtify) to apply it to various alg
"""
#mutiple models tobe built and check the accuracy, ROC, Precission etc error mewtrics to select one model
# gridsearch with CV

X_1=df.drop(['Geography','Exited'], axis=1)
X_1.shape #(9859, 9)
X_1.columns
tobestandardised = ['CreditScore',  'Age', 'Balance', 'EstimatedSalary']
StandardCols= [ 'Gender',  'Tenure',  'NumOfProducts', 'HasCrCard', 'IsActiveMember' ]
X_StandardCols = X_1[ StandardCols ]
X_tobestandardised = X_1[ tobestandardised ]

## Feature scaling Standardisatin and Normalistaion
#Standardisation
"""
code for Normalisation
"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc_model = sc.fit(X_tobestandardised)

X_tobestandardised= sc.transform(X_tobestandardised)
X_tobestandardised=pd.DataFrame(X_tobestandardised, columns= tobestandardised)


#Normalisation


df_Geo_getDummies # encode data frame[3 cols i.e  each for a factor] on Geography using pd.getdummies
df_fh_geo_1col  # encode data frame[1 col] on Geography using feature hasing from sklearn
df_geo_catcode  # encode data frame[3 cols i.e  each for a factor] on Geography using cat.codes on catogericall variable
df_geo_labelBinariser.shape #  # encode data frame[3 cols i.e  each for a factor] on Geography using labelbinariser from sklearn
#(9859, 3)

# Adding all  the features to a dataframe X
X=pd.concat([X_1.reset_index(drop=True), df_geo_labelBinariser],axis=1)
#X=pd.concat([X_StandardCols.reset_index(drop=True),X_tobestandardised, df_fh_geo_1col],axis=1)
X.shape #(9859, 12)

Y=df.loc[:,'Exited']
Y.shape # (9859,)
Y=Y.astype('int') ##if not converted to int type throws errors while fitting model


from collections import Counter
print('Imbalanced ratio in training set: 1:%i' % (Counter(Y)[0]/Counter(Y)[1]))

#splitting the data to train and test data
from sklearn.model_selection import train_test_split
#doing a stratified sample on Y i.e exited
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0,  stratify=Y)


## building  multiple models on diff algorithms
from sklearn.tree import DecisionTreeClassifier
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier

#Cross validation
from sklearn.grid_search import GridSearchCV
#from sklearn.model_selection import GridSearchCV
param_grid = {
    'criterion' :['entropy','gini'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth':[5,8,12,15],
    'min_samples_leaf':[5,8,12,15]
    
}

#rus = make_pipeline(RandomUnderSampler(),DecisionTreeClassifier(n_jobs=-1,random_state=5151))

### grid search for cart model
cart = DecisionTreeClassifier(random_state=5151)
CV_cart = GridSearchCV(estimator=cart, param_grid=param_grid, cv= 5)
CV_cart.fit(X_train,Y_train)

print (CV_cart.best_params_)
CV_cart.score(X_test, Y_test) #  0.8384043272481406


### grid search for Random forest model
rforest = RandomForestClassifier(n_jobs=-1,random_state=5151)
CV_rforest = GridSearchCV(estimator=rforest, param_grid=param_grid, cv= 5)
CV_rforest.fit(X_train,Y_train)
print (CV_rforest.best_params_)

print (CV_rforest.best_estimator_)
#best_params_
CV_rforest.score(X_test, Y_test) #  0.8542934415145369
"""
CV_rforest.grid_scores_
CV_rforest.grid_scores_[0][1]
CV_rforest.cv_validation_scores
type((CV_rforest.grid_scores_))
"""
for i in CV_rforest.grid_scores_:
    print(i.cv_validation_scores, type(i.cv_validation_scores))

for i in CV_rforest.grid_scores_:
    print(i.parameters)

type(CV_rforest.grid_scores_[1,1])


### grid search for Gradientboost model
param_grid = {
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth':[5,8,12,15],
    'min_samples_leaf':[5,8,12,15]
    
}
gboost = GradientBoostingClassifier(warm_start=True,random_state=5151)
CV_gboost = GridSearchCV(estimator=gboost, param_grid=param_grid, cv= 5)
CV_gboost.fit(X_train,Y_train)
print (CV_gboost.best_params_)
CV_gboost.score(X_test, Y_test)  # 0.8644354293441514


### grid search for KNN classifier model
from sklearn.neighbors import KNeighborsClassifier
param_grid = {
    'n_neighbors': [3,5,8,9,10,12]
    
}
KNN= KNeighborsClassifier( n_jobs=-1)
CV_KNN = GridSearchCV(estimator=KNN, param_grid=param_grid, cv= 5)
CV_KNN.fit(X_train,Y_train)
print (CV_KNN.best_params_)
CV_KNN.score(X_test, Y_test) #00.8275862068965517, 0.7914131169709263

### grid search for adaboost model
param_grid = {
    'learning_rate': [0.1,0.2,0.3,0.4,0.5]
   
}

adaboost = AdaBoostClassifier()
CV_adaoost = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv= 5)
CV_adaoost.fit(X_train,Y_train)
print (CV_adaoost.best_params_)
CV_adaoost.score(X_test, Y_test) #0.8559837728194726, 0.8559837728194726

### grid search for Bagging classifier model
param_grid = {
    'n_estimators': [10,20,30,40,50,60,70,80],
    'max_samples':[5,10,15,20]
    
}

Bagg = BaggingClassifier(warm_start=True,random_state=5151)
CV_Bagg = GridSearchCV(estimator=Bagg, param_grid=param_grid, cv= 5)
CV_Bagg.fit(X_train,Y_train)
print (CV_Bagg.best_params_)
CV_Bagg.score(X_test, Y_test) #0.8133874239350912, 0.8184584178498986


def roc_auc_plot(y_true, y_proba, y_pred,label=' ', l='-', lw=1.0):
    from sklearn.metrics import roc_curve, roc_auc_score, precision_score, f1_score
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f, prec=%.3f, F1=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1]), 
                       precision_score(y_true,y_pred), f1_score(y_true,y_pred) ))

f, ax = plt.subplots(figsize=(6,6))

roc_auc_plot(Y_test, CV_cart.predict_proba(X_test), CV_cart.predict(X_test), label='CART', l='-')
roc_auc_plot(Y_test, CV_rforest.predict_proba(X_test),CV_rforest.predict(X_test), label='RFOREST', l='--')
#roc_auc_plot(Y_test, CV_gboost.predict_proba(X_test),CV_gboost.predict(X_test), label='GBOOST', l='-.')
roc_auc_plot(Y_test, CV_adaoost.predict_proba(X_test),CV_adaoost.predict(X_test), label='ADABOOST', l=':')
roc_auc_plot(Y_test, CV_KNN.predict_proba(X_test),CV_KNN.predict(X_test), label='KNN', l='-')
roc_auc_plot(Y_test, CV_Bagg.predict_proba(X_test),CV_Bagg.predict(X_test), label='BAGG', l='--')


ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', label='Random Classifier')    
ax.legend(loc="lower right")    
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title('Receiver Operator Characteristic[ROC] curves')

## GBOOST is having the highest ROC and AUC score and also high accuracy

Gboost_model = GradientBoostingClassifier(max_depth= 5, max_features = 'sqrt', min_samples_leaf= 8,
                                          warm_start=True,random_state=5151)
Gboost_model.fit(X_train,Y_train)
Gboost_model.score(X_test, Y_test)

Gboost_model.save("model.pkl")
### persisting the required models
le # label encoder
lb #label binariser
fh_Tenure_1col #feature hashing
sc_model # standardisation
Gboost_model #

