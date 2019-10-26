import pandas as pd
import sklearn

df = pd.read_csv('Adult_train.tab', sep='\t', skiprows=[1,2])

# print(df.head())
# print(df.columns)
# print(df.dtypes)
# print(df.count())
# print(df.isna().sum())

# print(df['y'].values)



y_train = df['y']
x_train = df.drop(columns=['y', 'education'])

x_train = pd.get_dummies(x_train, prefix=['race', 'sex', 'wclass', 'marital', 'occ', 'rel', 'country'],
                         columns=['race', 'sex', 'workclass', 'marital-status', 'occupation', 'relationship', 'native-country'])

from sklearn.feature_extraction import FeatureHasher

# fh = FeatureHasher(n_features=8, input_type='string')
# sp = fh.fit_transform(x_train['native-country'])
# df1 = pd.DataFrame(sp.toarray(), columns=['country_fh1', 'country_fh2', 'country_fh3', 'country_fh4',
#                                           'country_fh5', 'country_fh6', 'country_fh7', 'country_fh8'])
# x_train_hashed = pd.concat([df1, x_train], axis=1).drop(columns=['native-country'])


# normalizowanie kolumn z danymi liczbowymi
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = pd.DataFrame(scaler.fit_transform(x_train))

from sklearn.model_selection import train_test_split


data_train, data_test, target_train, target_test = train_test_split(
    x_train, y_train, test_size=0.30, random_state=10)

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# svc_model = LinearSVC(random_state=0, max_iter=1000)
# pred_test = svc_model.fit(data_train, target_train).predict(data_test)
# pred_train = svc_model.fit(data_train, target_train).predict(data_train)
# print("LinearSVC accuracy : ", accuracy_score(target_test, pred_test, normalize=True))
# print(classification_report(y_true=target_test, y_pred=pred_test))
# print(classification_report(y_true=target_train, y_pred=pred_train))

# knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance')
# pred = knn_model.fit(data_train,target_train).predict(data_test)
#
# print("KNN accuracy : ", accuracy_score(target_test, pred, normalize=True))
# print(classification_report(y_true=target_test, y_pred=pred))
# print(confusion_matrix(y_true=target_test, y_pred=pred))

# print(x_train.dtypes)


# print(x_train.dtypes)

