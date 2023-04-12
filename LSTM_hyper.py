from requirements import *
#3. Popatrzenie na jakieś bardziej rozbudowane wersje (może dodanie hidden layerów)?
class SequenceDataset(Dataset):

    def __init__(self, df, target, features, sequence_length):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(df[target].values).float()
        self.X = torch.tensor(df[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

class LSTMClassification(nn.Module):

    def __init__(self, number_of_features, number_of_classes, units, dropout, activation_function):
        super(LSTMClassification, self).__init__()
        self.number_of_features = number_of_features-1
        self.number_of_classes = number_of_classes
        self.units = units
        self.dropout = dropout
        if(activation_function == "relu"):
            self.activation_function = nn.ReLU()
        else:
            self.activation_function = nn.Tanh()
        self.lstm = nn.LSTM(input_size=self.number_of_features, hidden_size=self.units,  batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.units, number_of_classes)

    def forward(self, x):
        x = self.activation_function(x)
        lstm_out, (h, c) = self.lstm(x)
        logits = self.fc(lstm_out[:,-1])
        scores = F.sigmoid(logits)
        return scores 

class LSTM_tuning():
    def __init__(self, train_lista, valid_lista, test_lista, number_of_folds, kwantyle, selekcja,  name, estimator, timesteps, num_trials, batch_size=16):
        self.nazwa_estymatora = "LSTM"
        self.train_lista = train_lista
        self.valid_lista = valid_lista
        self.test_lista = test_lista
        self.number_of_folds = number_of_folds
        self.kwantyle = kwantyle
        self.selekcja = selekcja
        self.name = name
        self.estimator = estimator
        self.timesteps = timesteps
        self.num_trials = num_trials
        self.batch_size = batch_size

    def fitting(self, train, units, dropout, activation_function, optimizer_name, learning_rate, number_of_epochs, test=None, save_model=False):
        self.save_model = save_model
        if(test == None):
            training_len = int(len(train)*0.75)
            data_train = train.iloc[:training_len]
            data_test = train.iloc[training_len:]
        else:
            data_train = train
            data_test = test
        #Ładowane tego do torcha
        train_dataset = SequenceDataset(df=data_train,target="TARGET",features=data_train.columns.difference(["TARGET"]), sequence_length=self.timesteps)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataset = SequenceDataset(df=data_test,target="TARGET",features=data_test.columns.difference(["TARGET"]), sequence_length=self.timesteps)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        model = LSTMClassification(number_of_features = len(data_train.columns), number_of_classes=1, units=units, dropout=dropout, activation_function=activation_function)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
        loss_function = nn.BCELoss()
        best_balanced_accuracy_test = 0
        for epoch in range(number_of_epochs):
            for train_index, train_data in enumerate(train_loader, 0):
                X_train, y_train = train_data
                model.zero_grad()
                output_probability_train = model(X_train)
                y_train = y_train.unsqueeze(1)
                train_loss = loss_function(output_probability_train, y_train.float())
                train_loss.backward()
                optimizer.step()
            with torch.no_grad():
                model.eval()
                number_of_positive_test = 0
                number_of_negative_test = 0
                true_positive_test = 0
                true_negative_test = 0
                for test_index, test_data in enumerate(test_loader, 0):
                    X_test, y_test = test_data
                    output_probability_test = model(X_test)
                    y_test = y_test.unsqueeze(1)
                    y_test_pred = (output_probability_test > 0.5).float()
                    number_of_positive_test = number_of_positive_test + np.sum([1 if i==1 else 0 for i in y_test])
                    number_of_negative_test = number_of_negative_test + np.sum([1 if i==0 else 0 for i in y_test])
                    true_positive_test = true_positive_test + np.sum([1 if i==j and i==1 else 0 for i, j in zip(y_test_pred, y_test)])
                    true_negative_test = true_negative_test + np.sum([1 if i==j and i==0 else 0 for i, j in zip(y_test_pred, y_test)])
                balanced_accuracy_test = (true_positive_test/number_of_positive_test + true_negative_test/number_of_negative_test)/2
                if(balanced_accuracy_test > best_balanced_accuracy_test):
                    best_balanced_accuracy_test = balanced_accuracy_test
                    if(self.save_model == True):
                        torch.save(model.state_dict(), 'saved_models/temp_best_model.pth')
                            #print(print("Weight saved: {}".format(model.fc.weight.detach().numpy()[-1][-1])))
        return best_balanced_accuracy_test

    def ewaluacja(self, model, test, save_predictions=False):
        test_dataset = SequenceDataset(df=test,target="TARGET",features=test.columns.difference(["TARGET"]), sequence_length=self.timesteps)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        number_of_positive_test = 0
        number_of_negative_test = 0
        true_positive_test = 0
        true_negative_test = 0
        predykcje = []
        for test_index, test_data in enumerate(test_loader, 0):
            X_test, y_test = test_data
            output_probability_test = model(X_test)
            y_test = y_test.unsqueeze(1)
            y_test_pred = (output_probability_test>0.5).float()
            if(save_predictions == True):
                predykcje = predykcje + y_test_pred.flatten().tolist()
            number_of_positive_test = number_of_positive_test + np.sum([1 if i==1 else 0 for i in y_test])
            number_of_negative_test = number_of_negative_test + np.sum([1 if i==0 else 0 for i in y_test])
            true_positive_test = true_positive_test + np.sum([1 if i==j and i==1 else 0 for i, j in zip(y_test_pred, y_test)])
            true_negative_test = true_negative_test + np.sum([1 if i==j and i==0 else 0 for i, j in zip(y_test_pred, y_test)])
        balanced_accuracy_test = (true_positive_test/number_of_positive_test + true_negative_test/number_of_negative_test)/2
        if(save_predictions==True):
            return balanced_accuracy_test, predykcje
        else:
            return balanced_accuracy_test

    def hyper_optuna(self):
        if(self.estimator == self.nazwa_estymatora and self.selekcja == "łączne"):
            def objective_cv(trial):
                scores = []
                for kwantyl in range(1, len(self.kwantyle)+1):
                    train = self.train_lista[kwantyl-1]
                    valid = self.valid_lista[kwantyl-1]
                    test = self.test_lista[kwantyl-1]
                    units = trial.suggest_int('units', 8, 128)
                    dropout = trial.suggest_float('dropout', 0.0, 0.4)
                    activation_function = trial.suggest_categorical("activation_function", ["relu", "tanh"])
                    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
                    learning_rate = trial.suggest_float("learning_rate", 0.0005, 0.1, log=True)
                    balanced_accuracy_inner_folds = []
                    for fold in range(1, self.number_of_folds + 1):
                        balanced_accuracy_inner_folds.append(self.fitting(train=train[fold], units=units, dropout=dropout, activation_function=activation_function, optimizer_name=optimizer_name, learning_rate=learning_rate, number_of_epochs=20, test=None, save_model=False))
                return np.mean(balanced_accuracy_inner_folds)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective_cv, n_trials=self.num_trials)
            study_df = study.trials_dataframe().sort_values(by="value", ascending=False)
            print(study_df)
            units_list = study_df.loc[:study_df.index[2], "params_units"].values
            dropout_list = study_df.loc[:study_df.index[2], "params_dropout"].values
            activation_function_list = study_df.loc[:study_df.index[2], "params_activation_function"].values
            optimizer_name_list = study_df.loc[:study_df.index[2], "params_optimizer"].values
            learning_rate_list = study_df.loc[:study_df.index[2], "params_learning_rate"].values
            best_mean_score = 0
            i = 0 
            while(i < len(units_list)):
                scores = []
                for kwantyl in range(1, len(self.kwantyle)+1):
                    train = self.train_lista[kwantyl-1]
                    valid = self.valid_lista[kwantyl-1]
                    test = self.test_lista[kwantyl-1]
                    for fold in range(1, self.number_of_folds+1):
                        valid_score = self.fitting(train=train[fold], units=units_list[i], dropout=dropout_list[i], activation_function=activation_function_list[i], optimizer_name=optimizer_name_list[i], learning_rate=learning_rate_list[i], number_of_epochs=20, test=None, save_model=True)
                        model = LSTMClassification(number_of_features=len(train[fold].columns), number_of_classes=1, units=units_list[i], dropout=dropout_list[i], activation_function=activation_function_list[i])
                        model.load_state_dict(torch.load('saved_models/temp_best_model.pth'))
                        #Jeżeli chcemy zobaczyć hiperparametry to wystarczy po prostu print(model)
                        scores.append(self.ewaluacja(model=model, test=valid[fold], save_predictions=False))
                if(np.mean(scores) > best_mean_score):
                    best_mean_score = np.mean(scores)
                    best_set = i
                i = i + 1
                prediction_kwantyl = []
            final_scores = []
            predictions = []
            for kwantyl in range(1, len(self.kwantyle)+1):
                train = self.train_lista[kwantyl-1]
                valid = self.valid_lista[kwantyl-1]
                test = self.test_lista[kwantyl-1]
                prediction_kwantyl = []
                for fold in range(1, self.number_of_folds+1):
                    training = pd.concat([train[fold], valid[fold]])
                    test_score = self.fitting(train=training, units=units_list[best_set], dropout=dropout_list[best_set], activation_function=activation_function_list[best_set], optimizer_name=optimizer_name_list[best_set], learning_rate=learning_rate_list[best_set], number_of_epochs=20, test=None, save_model=True)
                    globals()[f"model_{kwantyl}_{fold}"] = LSTMClassification(number_of_features=len(training.columns), number_of_classes=1, units=units_list[best_set], dropout=dropout_list[best_set], activation_function=activation_function_list[best_set])
                    globals()[f"model_{kwantyl}_{fold}"].load_state_dict(torch.load('saved_models/temp_best_model.pth'))
                    score, preds = self.ewaluacja(model=globals()[f"model_{kwantyl}_{fold}"], test=test[fold], save_predictions=True)
                    prediction_kwantyl.append(preds)
                    final_scores.append(score)
                predictions.append(prediction_kwantyl)
            return predictions
class Ewaluacja_LSTM():
    def __init__(self, df, train_lista, valid_lista, test_lista, number_of_folds, kwantyle, selekcja,  name, estimator, results, timesteps=15, num_trials=5, batch_size=16):
        self.df = df
        self.train_lista = train_lista
        self.valid_lista = valid_lista
        self.test_lista = test_lista
        self.number_of_folds = number_of_folds
        self.kwantyle = kwantyle
        self.selekcja = selekcja
        self.name = name
        self.estimator = estimator
        self.results = results
        self.timesteps = timesteps
        self.num_trials = num_trials
        self.batch_size = batch_size
        self.nazwa_estymatora = "LSTM"

    def temp_data_for_evaluation(self):
        if(self.estimator == self.nazwa_estymatora and self.selekcja == "łączne"):
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