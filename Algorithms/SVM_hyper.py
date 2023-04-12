from requirements import *
class SVM_tuning():
    def __init__(self, train_lista, valid_lista, test_lista, number_of_folds, kwantyle, selekcja,  name, estimator, num_trials):
        self.nazwa_estymatora="SVM"
        self.train_lista = train_lista
        self.valid_lista = valid_lista
        self.test_lista = test_lista
        self.number_of_folds = number_of_folds
        self.kwantyle = kwantyle
        self.selekcja = selekcja
        self.name = name
        self.estimator = estimator
        self.num_trials = num_trials
    def hyper_optuna(self):
       if(self.estimator == self.nazwa_estymatora):
            def objective_cv(trial):
                scores = []
                for kwantyl in range(1, len(self.kwantyle)+1):
                    train = self.train_lista[kwantyl-1]
                    valid = self.valid_lista[kwantyl-1]
                    test = self.test_lista[kwantyl-1]
                    cross_validation = TimeSeriesSplit(n_splits=5)
                    param = {
                        'C': trial.suggest_float('C', 0.01, 1),
                        'degree': trial.suggest_int('degree', 2, 5),
                        'gamma': trial.suggest_float('gamma', 0.001, 1),
                        "kernel": "rbf",
                        "random_state": 17
                    }
                    for fold in range(1, self.number_of_folds+1):
                        for train_idx, val_idx in cross_validation.split(train[fold]):
                            X_inner_train = train[fold].iloc[train_idx].drop("TARGET", axis=1)
                            y_inner_train = train[fold].iloc[train_idx]["TARGET"]
                            X_inner_valid = train[fold].iloc[val_idx].drop("TARGET", axis=1)
                            y_inner_valid = train[fold].iloc[val_idx]["TARGET"]
                            model = svm.SVC(**param)
                            model.fit(X_inner_train, y_inner_train)
                            preds = model.predict(X_inner_valid)
                            accuracy = balanced_accuracy_score(y_inner_valid, preds)
                            scores.append(accuracy)
                return np.mean(scores)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective_cv, n_trials=self.num_trials)
            study_df = study.trials_dataframe().sort_values(by="value", ascending=False)
            C_list = study_df.loc[:study_df.index[10], "params_C"].values
            degree_list = study_df.loc[:study_df.index[10], "params_degree"].values
            gamma_list = study_df.loc[:study_df.index[10], "params_gamma"].values
            i = 0
            best_mean_score = 0
            while(i < len(C_list)):
                scores = []
                for kwantyl in range(1, len(self.kwantyle)+1):
                    train = self.train_lista[kwantyl-1]
                    valid = self.valid_lista[kwantyl-1]
                    test = self.test_lista[kwantyl-1]
                    for fold in range(1, self.number_of_folds+1):
                        X_train = train[fold].drop("TARGET", axis=1)
                        y_train = train[fold]["TARGET"]
                        X_valid = valid[fold].drop("TARGET", axis=1)
                        y_valid = valid[fold]["TARGET"]  
                        model = svm.SVC(C=C_list[i], degree=degree_list[i], gamma=gamma_list[i], kernel="rbf", random_state=17)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_valid)
                        score = balanced_accuracy_score(y_valid, y_pred)
                        scores.append(score)
                if(np.mean(scores)>best_mean_score):
                    best_mean_score=np.mean(scores)
                    best_set = i
                i = i + 1
            final_scores = []
            predictions = []
            for kwantyl in range(1, len(self.kwantyle)+1):
                train = self.train_lista[kwantyl-1]
                valid = self.valid_lista[kwantyl-1]
                test = self.test_lista[kwantyl-1]
                prediction_kwantyl = []
                for fold in range(1, self.number_of_folds+1):
                    training = pd.concat([train[fold], valid[fold]])
                    X_train = training.drop("TARGET", axis=1)
                    y_train = training["TARGET"]
                    X_test = test[fold].drop("TARGET", axis=1)
                    y_test = test[fold]["TARGET"]
                    estimator_final = svm.SVC(C=C_list[best_set], degree=degree_list[best_set], gamma=gamma_list[best_set], kernel="rbf", random_state=17)
                    estimator_final.fit(X_train, y_train)
                    y_pred = estimator_final.predict(X_test).flatten().tolist()
                    prediction_kwantyl.append(y_pred)
                    final_scores.append(balanced_accuracy_score(y_test, y_pred))
                predictions.append(prediction_kwantyl)
            return predictions

class Ewaluacja_SVM():
    def __init__(self, df, train_lista, valid_lista, test_lista, results, number_of_folds, kwantyle, selekcja, name, estimator):
        self.df = df
        self.train_lista = train_lista
        self.valid_lista = valid_lista
        self.test_lista = test_lista
        self.results = results
        self.number_of_folds = number_of_folds
        self.kwantyle = kwantyle
        self.selekcja = selekcja
        self.name = name
        self.estimator = estimator
        self.nazwa_estymatora = "SVM"

    def temp_data_for_evaluation(self):
        if(self.nazwa_estymatora == self.estimator):
            i = 0
            annualized_return_compounded = []
            annualized_standard_deviation = []
            maximum_drawdown = []
            information_ratio = []
            best_information_ratio = -np.inf
            while(i < len(self.kwantyle)):
                print("Kwantyl {}".format(self.kwantyle[i]))
                btscv = BlockingTimeSeriesSplit_with_valid(n_splits=self.number_of_folds)
                lista_test = []
                equity_folds = []
                fold = 0
                for train, valid, test in btscv.split(self.df):
                    df_test = self.df.iloc[test].copy()
                    predykcje = list(self.results[i][fold])
                    df_test['Predictions'] = predykcje
                    df_test["TARGET"] = df_test["TARGET"].shift(1)
                    df_test["Predictions"] = df_test["Predictions"].shift(1)
                    df_test.dropna(inplace=True)
                    lista_test.append(df_test)
                    fold = fold + 1
                df_temp = pd.concat([lista_test[0], lista_test[1], lista_test[2]])
                df_temp  = df_temp[['Price_Change_pct', 'Predictions', 'TARGET']]
                #Konwersja, aby ewaluacja była możliwa
                df_temp["TARGET"] = df_temp["TARGET"].apply(lambda x: -1 if x == 0 else x)
                df_temp["Predictions"] = df_temp["Predictions"].apply(lambda x: -1 if x == 0 else x)
                df_temp["Price_change_pct_predicted"] = df_temp.apply(lambda x: np.abs(x.Price_Change_pct) if x.TARGET==x.Predictions else -1*np.abs(x.Price_Change_pct), axis=1)
                y_true = df_temp["TARGET"]
                df_temp.dropna(inplace=True)
                df_temp["kasa"] = np.nan
                df_temp["kasa"][0] = 1000
                df_temp.fillna(0, inplace=True)
                df_temp["Wynik_final"] = self.compund_interest(price_change=df_temp["Price_Change_pct"].values, pozycje=df_temp["Predictions"].values, kasa = df_temp["kasa"])
                equity_folds.append(df_temp.loc[:"2019-09-01", "Wynik_final"].values)
                equity_folds.append(df_temp.loc["2019-09-01":"2021-05-02", "Wynik_final"].values)
                equity_folds.append(df_temp.loc["2021-05-02":, "Wynik_final"].values)
                print("Balanced accuracy score: {}".format(np.round(balanced_accuracy_score(y_true, df_temp["Predictions"].values), 4)))
                N = 1
                annualized_return_compounded.append(f'{round(self.annualized_return_compunded(wynik = df_temp["Wynik_final"], N = N), 4)*100}%')
                annualized_standard_deviation.append(f'{round(self.annualized_standard_deviation(price_change_predicted=df_temp["Price_change_pct_predicted"].values, N=N), 4)*100}%')
                maximum_drawdown.append(f'{round(self.maximum_drawdown(wynik = df_temp["Wynik_final"]), 4)*100}%')
                information_ratio.append(f'{round(self.information_ratio(wynik = df_temp["Wynik_final"], price_change_predicted = df_temp["Price_change_pct_predicted"].values, N = N), 4)*100}%')
                print("Annualized Return Compounded: {}%".format(np.round(self.annualized_return_compunded(wynik = df_temp["Wynik_final"], N = N), 4)*100))
                print("Annualized Standard Deviation: {}%".format(np.round(self.annualized_standard_deviation(price_change_predicted=df_temp["Price_change_pct_predicted"].values, N=N), 4)*100))
                print("Maximum drawdown: {}%".format(np.round(self.maximum_drawdown(wynik = df_temp["Wynik_final"]), 4)*100))
                print("Information Ratio: {}%".format(np.round(self.information_ratio(wynik = df_temp["Wynik_final"], price_change_predicted = df_temp["Price_change_pct_predicted"].values, N = N), 4)*100))
                if(float(information_ratio[i].replace("%",""))> best_information_ratio):
                    best_information_ratio = float(information_ratio[i].replace("%",""))
                    best_kwartyl = i+1
                    best_kwartyl_plot_data = equity_folds
                i = i + 1
            wyniki = pd.DataFrame(list(zip(annualized_return_compounded, annualized_standard_deviation, maximum_drawdown, information_ratio)), columns=["Annualized Return Compounded", "Annualized Standard Deviation", "Maximum drawdown", "Information Ratio"])
            with pd.ExcelWriter('Data/Wyniki.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                wyniki.to_excel(writer, sheet_name="{}_{}_{}".format(self.name, self.nazwa_estymatora, self.selekcja))
            return best_kwartyl_plot_data, best_kwartyl
            
    
    def accuracy(self, y_true, y_pred):
        correct = 0
        for true, pred in zip(y_true, y_pred):
            if(true == pred):
                correct = correct + 1
        return correct/len(y_true)

    def compund_interest(self, price_change, pozycje, kasa, fee=0.001):
        wartosc_portfela = np.zeros(price_change.shape)
        wartosc_portfela[0] = kasa[0]*(1+price_change[0]*pozycje[0])
        for i in range(1, wartosc_portfela.shape[0]):
            wartosc_portfela[i] = wartosc_portfela[i-1]*(1+price_change[i]*pozycje[i] - np.abs(pozycje[i] - pozycje[i-1])*fee)
        return wartosc_portfela

    def annualized_return_compunded(self, wynik, N):
        return (wynik[-1]/wynik[0])**(1/N)-1
    
    def annualized_standard_deviation(self, price_change_predicted, N):
        return np.sqrt(N*np.sum((price_change_predicted-np.mean(price_change_predicted))**2))
    
    def sharpe_ratio(self, wynik, kolumna_ze_zwrotem, N):
        return max(self.annualized_return_compunded(wynik, N)/(kolumna_ze_zwrotem.std()*(N*365)**0.5), 0)

    def maximum_drawdown(self, wynik):
        maxDif = 0
        start = wynik[0]
        drawdown_i = 0
        for i in range(len(wynik)):
            if(maxDif != min(maxDif, wynik[i]-start)):
                worst_i = i
            maxDif = min(maxDif, wynik[i]-start)
            if(wynik[i] > start):
                drawdown_i = i
            start = max(wynik[i], start)
        return -(wynik[worst_i]-wynik[drawdown_i])/wynik[drawdown_i]
    
    def calmar_ratio(self, wynik, N):
        return self.annualized_return_compunded(wynik, N)/self.maximum_drawdown(wynik)
    
    def information_ratio(self, wynik, price_change_predicted, N):
        return self.annualized_return_compunded(wynik=wynik, N=N)/self.annualized_standard_deviation(price_change_predicted=price_change_predicted, N=N)