from requirements import *
class Ewaluacja_BaseModel():
    def __init__(self, df, name, number_of_folds):
        self.df = df
        self.name = name
        self.number_of_folds = number_of_folds

    def temp_data_for_evaluation(self):
        btscv = BlockingTimeSeriesSplit_with_valid(n_splits=self.number_of_folds)
        lista_test = []
        equity_folds = []
        annualized_return_compounded = []
        annualized_standard_deviation = []
        maximum_drawdown = []
        information_ratio = []
        for train, valid, test in btscv.split(self.df):
            df_test = self.df.iloc[test].copy()
            df_test['Predictions'] = df_test["PREV_TARGET"]
            df_test["Predictions"] = df_test["Predictions"].shift(1)
            df_test["TARGET"] = df_test["TARGET"].shift(1)
            df_test.dropna(inplace=True)
            lista_test.append(df_test)
        df_temp = pd.concat([lista_test[0], lista_test[1], lista_test[2]])
        y_true = df_temp["TARGET"]
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
        wyniki = pd.DataFrame(list(zip(annualized_return_compounded, annualized_standard_deviation, maximum_drawdown, information_ratio)), columns=["Annualized Return Compounded", "Annualized Standard Deviation", "Maximum drawdown", "Information Ratio"])
        with pd.ExcelWriter('Data/Wyniki.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            wyniki.to_excel(writer, sheet_name="{}_{}".format(self.name, "Base"))
        return equity_folds
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