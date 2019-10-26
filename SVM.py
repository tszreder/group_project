from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from data_load import x_train, y_train, data_train, data_test, target_train, target_test
from sklearn.model_selection import GridSearchCV


tuned_parameters = {'C': [0.1, 1, 10, 100]}

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(LinearSVC(), tuned_parameters, cv=2,
                       scoring='%s_macro' % score)
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     # y_true, y_pred = target_valid, clf.predict(target_train1)
#     # print(classification_report(y_true, y_pred))
#     # print()

svc_model = LinearSVC(random_state=0, max_iter=1000, C=1000)
pred_test = svc_model.fit(data_train, target_train).predict(data_test)
print("SVC accuracy, C=1000 : ", accuracy_score(target_test, pred_test, normalize=True))
# print("SVC recall, C=1000 : ", recall_score(target_test, pred_test))
# print(classification_report(y_true=target_test, y_pred=pred_test))

svc_model = LinearSVC(random_state=0, max_iter=1000, C=1)
pred_test = svc_model.fit(data_train, target_train).predict(data_test)
print("SVC accuracy, C=1 : ", accuracy_score(target_test, pred_test, normalize=True))
# print("SVC recall, C=1 : ", recall_score(target_test, pred_test))
# print(classification_report(y_true=target_test, y_pred=pred_test))

svc_model = LinearSVC(random_state=0, max_iter=1000, C=0.1)
pred_test = svc_model.fit(data_train, target_train).predict(data_test)
print("SVC accuracy, C=0.1 : ", accuracy_score(target_test, pred_test, normalize=True))
# print("SVC recall, C=1 : ", recall_score(target_test, pred_test))
# print(classification_report(y_true=target_test, y_pred=pred_test))