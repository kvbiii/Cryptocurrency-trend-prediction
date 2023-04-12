from requirements import *
from prepare_data import *
from Feature_selection import *
from LGBM_hyper import *
from Logistic_hyper import *
from  RandomForest_hyper import *
from SVM_hyper import *
from LSTM_hyper import *
from Base_model import *
from plots import *

def main(df, selekcja, name, number_of_folds, lista_kwantyli, estimator, timesteps, num_trials):
    train_dictionary, valid_dictionary, test_dictionary = Prepare_data(df=df, number_of_folds=number_of_folds).feature_extension()
    train_list, valid_list, test_list = Recursive_feature_selection_CV(train=train_dictionary, valid=valid_dictionary, test=test_dictionary, number_of_folds=number_of_folds, kwantyle=lista_kwantyli, estimator=estimator, selekcja=selekcja).chosen_columns()
    predictions_lgbm = LGBM_tuning(train_lista=train_list, valid_lista=valid_list, test_lista=test_list, number_of_folds=number_of_folds, kwantyle=lista_kwantyli, selekcja=selekcja,  name=name, estimator=estimator, num_trials=num_trials).hyper_optuna()
    try:
        results, kwartyl = Ewaluacja_LGBM(df=df, train_lista=train_list, valid_lista=valid_list, test_lista=test_list, results=predictions_lgbm, number_of_folds=number_of_folds, kwantyle=lista_kwantyli, selekcja=selekcja, name=name, estimator=estimator).temp_data_for_evaluation()
    except:
        pass
    predictions_logistic = Logistic_tuning(train_lista=train_list, valid_lista=valid_list, test_lista=test_list, number_of_folds=number_of_folds, kwantyle=lista_kwantyli, selekcja=selekcja,  name=name, estimator=estimator, num_trials=num_trials).hyper_optuna()
    try:
        results, kwartyl = Ewaluacja_Logistic(df=df, train_lista=train_list, valid_lista=valid_list, test_lista=test_list, results=predictions_logistic, number_of_folds=number_of_folds, kwantyle=lista_kwantyli, selekcja=selekcja, name=name, estimator=estimator).temp_data_for_evaluation()
    except:
        pass
    predictions_svm = SVM_tuning(train_lista=train_list, valid_lista=valid_list, test_lista=test_list, number_of_folds=number_of_folds, kwantyle=lista_kwantyli, selekcja=selekcja,  name=name, estimator=estimator, num_trials=num_trials).hyper_optuna()
    try:
        results, kwartyl = Ewaluacja_SVM(df=df, train_lista=train_list, valid_lista=valid_list, test_lista=test_list, results=predictions_svm, number_of_folds=number_of_folds, kwantyle=lista_kwantyli, selekcja=selekcja, name=name, estimator=estimator).temp_data_for_evaluation()
    except:
        pass
    predictions_forest = RandomForest_tuning(train_lista=train_list, valid_lista=valid_list, test_lista=test_list, number_of_folds=number_of_folds, kwantyle=lista_kwantyli, selekcja=selekcja,  name=name, estimator=estimator, num_trials=num_trials).hyper_optuna()
    try:
        results, kwartyl = Ewaluacja_Forest(df=df, train_lista=train_list, valid_lista=valid_list, test_lista=test_list, results=predictions_forest, number_of_folds=number_of_folds, kwantyle=lista_kwantyli, selekcja=selekcja, name=name, estimator=estimator).temp_data_for_evaluation()
    except:
        pass
    predictions_lstm = LSTM_tuning(train_lista=train_list, valid_lista=valid_list, test_lista=test_list, number_of_folds=number_of_folds, kwantyle=lista_kwantyli, selekcja=selekcja,  name=name, estimator=estimator, timesteps=timesteps, num_trials=5, batch_size=16).hyper_optuna()
    try:
        results, kwartyl = Ewaluacja_LSTM(df=df, train_lista=train_list, valid_lista=valid_list, test_lista=test_list, results=predictions_lstm, number_of_folds=number_of_folds, kwantyle=lista_kwantyli, selekcja=selekcja,  name=name, estimator=estimator, timesteps=15, num_trials=5, batch_size=16).temp_data_for_evaluation()
    except:
        pass
    results_base = Ewaluacja_BaseModel(df=df, name=name, number_of_folds=number_of_folds).temp_data_for_evaluation()
    return results, kwartyl, results_base
if __name__ == '__main__':
    kryptos = ["bitcoin", "bitcoin_dodatek", "ethereum", "ethereum_dodatek"]
    estimators = ["RF", "LR", "SVM", "LGBM", "LSTM"]
    for krypto in kryptos:
        print(krypto)
        for estimator in estimators:
            print(estimator)
            results_łączne, kwartyl_łączne, results_base=main(df=pd.read_csv("Data/licencjat_{}_7.csv".format(krypto), index_col=0), selekcja="łączne", name=krypto, number_of_folds=3, lista_kwantyli=[25, 50, 75, 100], estimator=estimator, timesteps = 30, num_trials=12)
            if(estimator != "LSTM"):
                results_pojedynczy, kwartyl_pojedynczy, results_base=main(df=pd.read_csv("Data/licencjat_{}_7.csv".format(krypto), index_col=0), selekcja="pojedyncze", name=krypto, number_of_folds=3, lista_kwantyli=[25, 50, 75, 100], estimator=estimator, timesteps = 30, num_trials=20)
                plotting(results_pojedynczy=results_pojedynczy, kwartyl_pojedynczy=kwartyl_pojedynczy, results_łączne=results_łączne, kwartyl_łączne=kwartyl_łączne, results_base=results_base, name=krypto, nazwa_estymatora=estimator)
            else:
                plotting(results_pojedynczy=None, kwartyl_pojedynczy=None, results_łączne=results_łączne, kwartyl_łączne=kwartyl_łączne, results_base=results_base, name=krypto, nazwa_estymatora=estimator)