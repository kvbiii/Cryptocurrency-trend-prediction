from requirements import *
class Recursive_feature_selection_CV():
    def __init__(self, train, valid, test, number_of_folds, kwantyle, estimator, selekcja):
        self.train = train
        self.valid = valid
        self.test = test
        self.number_of_folds = number_of_folds
        self.kwantyle = kwantyle
        self.estimator = estimator
        self.selekcja=selekcja
        if(self.selekcja != "łączne"):
            if(self.estimator == "LGBM"):
                self.model = LGBMClassifier(random_state=17)
            elif(self.estimator == "LR"):
                self.model = LogisticRegression(random_state=17, max_iter=5000)
            elif(self.estimator == "SVM"):
                self.model = svm.SVC(kernel="linear", random_state=17)
            else:
                self.model = RandomForestClassifier(random_state=17)
    def selecting(self):
        selected_features = []
        for fold in range(1, self.number_of_folds+1):
            X_train = self.train[fold].drop("TARGET", axis=1)
            y_train = self.train[fold]["TARGET"]
            estimator1 = LGBMClassifier(random_state=17)
            estimator2 = LogisticRegression(random_state=17)
            estimator3 = svm.SVC(kernel="linear", random_state=17)
            estimator4 = RandomForestClassifier(random_state=17)
            folds = TimeSeriesSplit(n_splits=5)

            if(self.selekcja == "łączne"):
                rfe_lgbm = RFECV(estimator=estimator1, step=1, cv=folds, verbose=0, min_features_to_select=1, importance_getter="auto", scoring='roc_auc')
                rfe_lgbm.fit(X_train, y_train)
                selected_features.append(rfe_lgbm.ranking_)

                rfe_logistic = RFECV(estimator=estimator2, step=1, cv=folds, verbose=0, min_features_to_select=1, importance_getter="auto", scoring='roc_auc')
                rfe_logistic.fit(X_train, y_train)
                selected_features.append(rfe_logistic.ranking_)

                rfe_svc = RFECV(estimator=estimator3, step=1, cv=folds, verbose=0, min_features_to_select=1, importance_getter="auto", scoring='roc_auc')
                rfe_svc.fit(X_train, y_train)
                selected_features.append(rfe_svc.ranking_)

                rfe_forest = RFECV(estimator=estimator4, step=1, cv=folds, verbose=0, min_features_to_select=1, importance_getter="auto", scoring='roc_auc')
                rfe_forest.fit(X_train, y_train)
                selected_features.append(rfe_forest.ranking_)
            else:
                rfe = RFECV(estimator=self.model, step=1, cv=folds, verbose=0, min_features_to_select=1, importance_getter="auto", scoring='roc_auc')
                rfe.fit(X_train, y_train)
                selected_features.append(rfe.ranking_)

        return np.sum(selected_features, axis=0)

    def chosen_columns(self):
        features_sum = self.selecting()
        train_list = []
        valid_list = []
        test_list = [] 
        for kwantyl in range(1, len(self.kwantyle)+1):
            folds = [i for i in range(1, self.number_of_folds + 1)]
            train_final = dict(zip(folds, [None]*len(folds)))
            valid_final = dict(zip(folds, [None]*len(folds)))
            test_final = dict(zip(folds, [None]*len(folds)))
            final_features = np.where(features_sum <= np.quantile(features_sum, self.kwantyle[kwantyl-1]/100), 1, 0)
            feature_names = self.train[1].columns
            new_features = []
            for i, feature in zip(final_features, feature_names):
                if(i==1):
                    new_features.append(feature)
            new_features.append("TARGET")
            for fold in range(1, self.number_of_folds+1):
                train_final[fold] = self.train[fold][new_features]
                valid_final[fold] = self.valid[fold][new_features]
                test_final[fold] = self.test[fold][new_features]
            train_list.append(train_final)
            valid_list.append(valid_final)
            test_list.append(test_final)
        return train_list, valid_list, test_list