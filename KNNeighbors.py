from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from data_load import x_train, y_train, data_train, data_test, target_train, target_test
from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'n_neighbors': [3, 4, 5]}, {'weights': ['uniform', 'distance']}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=2,
                       scoring='%s_macro' % score)
    clf.fit(x_train, y_train)