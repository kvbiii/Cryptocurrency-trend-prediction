from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
class Prepare_data():
  def __init__(self, df, number_of_folds):
    self.df = df
    self.number_of_folds = number_of_folds
    self.scaler = StandardScaler()
  def splitting_data_into_folds(self):
    btscv = BlockingTimeSeriesSplit_with_valid(n_splits=self.number_of_folds)
    folds = [i for i in range(1, self.number_of_folds + 1)]
    train_dict = dict(zip(folds, [None]*len(folds)))
    valid_dict = dict(zip(folds, [None]*len(folds)))
    test_dict = dict(zip(folds, [None]*len(folds)))
    fold = 1
    for train, valid, test in btscv.split(self.df):
      #print("{} block train: {} - {} (length of train: {}) valid: {} - {} (length of valid: {}) test: {} - {} (length of test: {})".format(fold, self.df.iloc[train].index[0],self.df.iloc[train].index[-1], len(train), self.df.iloc[valid].index[0],self.df.iloc[valid].index[-1], len(valid), self.df.iloc[test].index[0],self.df.iloc[test].index[-1], len(test)))
      train_dict[fold] = self.df.iloc[train]
      valid_dict[fold] = self.df.iloc[valid]
      test_dict[fold] = self.df.iloc[test]
      fold = fold + 1
    return train_dict, valid_dict, test_dict
  def feature_extension(self):
    train, valid, test = self.splitting_data_into_folds()
    for fold in range(1, self.number_of_folds+1):
      scaler = self.scaler
      i = 0
      pierwotna_liczba_kolumn = len(train[fold].columns)
      kolumny_do_usuniecia = []
      while(i < pierwotna_liczba_kolumn):
        nazwa_kolumny = train[fold].columns[i]
        if (nazwa_kolumny not in ("TARGET", "PREV_TARGET")):
          train[fold][f"{nazwa_kolumny}_scaled"] = scaler.fit_transform(train[fold][[nazwa_kolumny]])
          valid[fold][f"{nazwa_kolumny}_scaled"] = scaler.transform(valid[fold][[nazwa_kolumny]])
          test[fold][f"{nazwa_kolumny}_scaled"] = scaler.transform(test[fold][[nazwa_kolumny]])
          kolumny_do_usuniecia.append(nazwa_kolumny)
        i = i + 1
      train[fold].drop(kolumny_do_usuniecia, axis=1, inplace=True)
      valid[fold].drop(kolumny_do_usuniecia, axis=1, inplace=True)
      test[fold].drop(kolumny_do_usuniecia, axis=1, inplace=True)
      train[fold].insert(int(len(train[fold].columns)-1), 'TARGET', train[fold].pop('TARGET'))
      valid[fold].insert(int(len(valid[fold].columns)-1), 'TARGET', valid[fold].pop('TARGET'))
      test[fold].insert(int(len(test[fold].columns)-1), 'TARGET', test[fold].pop('TARGET'))
    return train, valid, test
