import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, r2_score, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import joblib
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from skopt import BayesSearchCV

class Model:
    def __init__(self, df):
        self.df = df.copy()
        self.df.drop(columns = ["tmax_mean"], inplace = True)
        self.df.drop(columns = ["tmin_mean"], inplace = True)
        self.df.drop(columns = ["temp_prom"], inplace = True)
        self.df.drop(columns = ["precip_total"], inplace = True)
        self.df.drop(columns = ["precip_max"], inplace = True)
        self.df.drop(columns = ["precip_min"], inplace = True)
        self.df.drop(columns = ["precip_median"], inplace = True)
        self.df.drop(columns = ["precip_mean"], inplace = True)
        self.df.drop(columns = ["ndvi"], inplace = True)
        self.df.drop(columns = ["aedes_aegypti"], inplace = True)
        self.df.drop(columns = ["criaderos_de_mosquito"], inplace = True)
        self.df.drop(columns = ["fiebre"], inplace = True)
        self.df.drop(columns = ["plaquetas"], inplace = True)
        self.df.drop(columns = ["buscar_trabajo"], inplace = True)
        self.df.drop(columns = ["desempleo"], inplace = True)
        self.df.drop(columns = ["empleo"], inplace = True)
        self.df.drop(columns = ["comprar_carro"], inplace = True)
        self.df.drop(columns = ["préstamo"], inplace = True)
        self.df.drop(columns = ["tiquetes_avion"], inplace = True)
        self.df.drop(columns = ["vacaciones"], inplace = True)
        self.df.drop(columns = ["corte_agua"], inplace = True)
        self.df.drop(columns = ["corte_de_agua"], inplace = True)
        self.df.drop(columns = ["suspension_de_agua"], inplace = True)
        self.df.drop(columns = ["dengue"], inplace = True)

    def drop_lags(self, i, f, df_temp):
        for i in range(i, f):
            df_temp.drop(columns = [f"tmax_mean_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"tmin_mean_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"temp_prom_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"precip_total_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"precip_max_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"precip_min_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"precip_median_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"precip_mean_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"ndvi_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"nino34ssta_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"aedes_aegypti_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"criaderos_de_mosquito_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"fiebre_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"plaquetas_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"buscar_trabajo_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"desempleo_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"empleo_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"comprar_carro_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"préstamo_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"tiquetes_avion_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"vacaciones_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"corte_agua_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"corte_de_agua_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"suspension_de_agua_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"dengue_lag_{i}"], inplace = True)
        return df_temp
    
    def partition_cases(self, momento):
        df_temp = self.df.copy()

        df_temp.drop(columns=['clasi_rr'], inplace=True)
        df_temp.drop(columns = ["rr"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0"], inplace = True)

        if momento == "inmediato":
            df_temp = self.drop_lags(i = 5, f = 9, df_temp = df_temp)
            for i in range(5, 9):
                df_temp.drop(columns = [f"casos_lag_{i}"], inplace = True)

        elif momento == "mediano":
            df_temp = self.drop_lags(i = 1, f = 5, df_temp = df_temp)
            for i in range(1, 5):
                df_temp.drop(columns = [f"casos_lag_{i}"], inplace = True)
        elif momento == "todos":
            pass
        else:
            print("Error in partition_class: momento is not valid")

        df_temp.drop(columns = ["rr_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_1"], inplace = True)
        df_temp.drop(columns = ["rr_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_2"], inplace = True)
        df_temp.drop(columns = ["rr_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_3"], inplace = True)
        df_temp.drop(columns = ["rr_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_4"], inplace = True)
        df_temp.drop(columns = ["rr_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_5"], inplace = True)
        df_temp.drop(columns = ["rr_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_6"], inplace = True)
        df_temp.drop(columns = ["rr_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_7"], inplace = True)
        df_temp.drop(columns = ["rr_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_8"], inplace = True)

        df_temp = pd.get_dummies(df_temp, columns = ["urb"])

        train = df_temp[df_temp['week_canton'] < '2024-1-101']
        val = df_temp[(df_temp['week_canton'] >= '2024-1-101') & (df_temp['week_canton'] < '2025-1-101')]
        test = df_temp[df_temp['week_canton'] >= '2025-1-101']

        X_train = train.drop(columns = ["casos", "week_canton"])
        y_train = train["casos"]

        X_val = val.drop(columns = ["casos", "week_canton"])
        y_val= val["casos"] 

        X_test= test.drop(columns = ["casos", "week_canton"])
        y_test = test["casos"] 

        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)

        split_indices = np.zeros(len(X_combined))
        split_indices[:len(X_train)] = -1
        split_indices[len(X_train):] = 0

        pds = PredefinedSplit(test_fold=split_indices)

        scaler = StandardScaler()  

        X_combined_scaled = pd.DataFrame(scaler.fit_transform(X_combined), columns=X_combined.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        return X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds

    def partition_rr(self, momento):
        df_temp = self.df.copy()

        df_temp.drop(columns=['clasi_rr'], inplace=True)
        df_temp.drop(columns = ["casos"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0"], inplace = True)

        if momento == "inmediato":
            df_temp = self.drop_lags(i = 5, f = 9, df_temp = df_temp)
            for i in range(5, 9):
                df_temp.drop(columns = [f"rr_lag_{i}"], inplace = True)

        elif momento == "mediano":
            df_temp = self.drop_lags(i = 1, f = 5, df_temp = df_temp)
            for i in range(1, 5):
                df_temp.drop(columns = [f"rr_lag_{i}"], inplace = True)
        elif momento == "todos":
            pass
        else:
            print("Error in partition_class: momento is not valid")

        df_temp.drop(columns = ["casos_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_1"], inplace = True)
        df_temp.drop(columns = ["casos_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_2"], inplace = True)
        df_temp.drop(columns = ["casos_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_3"], inplace = True)
        df_temp.drop(columns = ["casos_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_4"], inplace = True)
        df_temp.drop(columns = ["casos_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_5"], inplace = True)
        df_temp.drop(columns = ["casos_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_6"], inplace = True)
        df_temp.drop(columns = ["casos_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_7"], inplace = True)
        df_temp.drop(columns = ["casos_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_8"], inplace = True)

        df_temp = pd.get_dummies(df_temp, columns = ["urb"])

        train = df_temp[df_temp['week_canton'] < '2024-1-101']
        val = df_temp[(df_temp['week_canton'] >= '2024-1-101') & (df_temp['week_canton'] < '2025-1-101')]
        test = df_temp[df_temp['week_canton'] >= '2025-1-101']

        X_train = train.drop(columns = ["rr", "week_canton"])
        y_train = train["rr"]

        X_val = val.drop(columns = ["rr", "week_canton"])
        y_val= val["rr"]

        X_test= test.drop(columns = ["rr", "week_canton"])
        y_test = test["rr"]

        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)

        split_indices = np.zeros(len(X_combined))
        split_indices[:len(X_train)] = -1
        split_indices[len(X_train):] = 0

        pds = PredefinedSplit(test_fold=split_indices)

        scaler = StandardScaler()  

        X_combined_scaled = pd.DataFrame(scaler.fit_transform(X_combined), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        X_test_for_reg = pd.concat([X_val, X_test], axis=0)

        return X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds

    def partition_classi_rr(self, momento):
        df_temp = self.df.copy()

        df_temp.drop(columns=['casos'], inplace=True)
        df_temp.drop(columns = ["rr"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0"], inplace = True)

        if momento == "inmediato":
            df_temp = self.drop_lags(i = 5, f = 9, df_temp = df_temp)
            for i in range(5, 9):
                df_temp.drop(columns = [f"clasi_rr_lag_{i}"], inplace = True)

            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_1"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_2"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_3"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_4"])

        elif momento == "mediano":
            df_temp = self.drop_lags(i = 1, f = 5, df_temp = df_temp)
            for i in range(1, 5):
                df_temp.drop(columns = [f"clasi_rr_lag_{i}"], inplace = True)

            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_5"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_6"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_7"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_8"])

        elif momento == "todos":
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_1"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_2"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_3"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_4"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_5"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_6"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_7"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_lag_8"])
        else:
            print("Error in partition_class: momento is not valid")

        df_temp.drop(columns = ["rr_lag_1"], inplace = True)
        df_temp.drop(columns = ["casos_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_1"], inplace = True)
        df_temp.drop(columns = ["rr_lag_2"], inplace = True)
        df_temp.drop(columns = ["casos_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_2"], inplace = True)
        df_temp.drop(columns = ["rr_lag_3"], inplace = True)
        df_temp.drop(columns = ["casos_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_3"], inplace = True)
        df_temp.drop(columns = ["rr_lag_4"], inplace = True)
        df_temp.drop(columns = ["casos_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_4"], inplace = True)
        df_temp.drop(columns = ["rr_lag_5"], inplace = True)
        df_temp.drop(columns = ["casos_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_5"], inplace = True)
        df_temp.drop(columns = ["rr_lag_6"], inplace = True)
        df_temp.drop(columns = ["casos_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_6"], inplace = True)
        df_temp.drop(columns = ["rr_lag_7"], inplace = True)
        df_temp.drop(columns = ["casos_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_7"], inplace = True)
        df_temp.drop(columns = ["rr_lag_8"], inplace = True)
        df_temp.drop(columns = ["casos_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_8"], inplace = True)

        df_temp = pd.get_dummies(df_temp, columns = ["urb"])

        train = df_temp[df_temp['week_canton'] < '2024-1-101']
        val = df_temp[(df_temp['week_canton'] >= '2024-1-101') & (df_temp['week_canton'] < '2025-1-101')]
        test = df_temp[df_temp['week_canton'] >= '2025-1-101']

        X_train = train.drop(columns = ["clasi_rr", "week_canton"])
        y_train = train["clasi_rr"]

        X_val = val.drop(columns = ["clasi_rr", "week_canton"])
        y_val= val["clasi_rr"]

        X_test= test.drop(columns = ["clasi_rr", "week_canton"])
        y_test = test["clasi_rr"]

        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)

        split_indices = np.zeros(len(X_combined))
        split_indices[:len(X_train)] = -1
        split_indices[len(X_train):] = 0

        pds = PredefinedSplit(test_fold=split_indices)

        scaler = StandardScaler()  

        X_combined_scaled = pd.DataFrame(scaler.fit_transform(X_combined), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        return X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds

    def partition_classi_rr_no_0(self, momento):
        df_temp = self.df.copy()

        df_temp.drop(columns=['casos'], inplace=True)
        df_temp.drop(columns = ["rr"], inplace = True)
        df_temp.drop(columns = ["clasi_rr"], inplace = True)

        if momento == "inmediato":
            df_temp = self.drop_lags(i = 5, f = 9, df_temp = df_temp)
            for i in range(5, 9):
                df_temp.drop(columns = [f"clasi_rr_no_0_lag_{i}"], inplace = True)

            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_1"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_2"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_3"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_4"])

        elif momento == "mediano":
            df_temp = self.drop_lags(i = 1, f = 5, df_temp = df_temp)
            for i in range(1, 5):
                df_temp.drop(columns = [f"clasi_rr_no_0_lag_{i}"], inplace = True)

            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_5"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_6"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_7"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_8"])

        elif momento == "todos":
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_1"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_2"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_3"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_4"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_5"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_6"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_7"])
            df_temp = pd.get_dummies(df_temp, columns = ["clasi_rr_no_0_lag_8"])
        else:
            print("Error in partition_class: momento is not valid")

        df_temp.drop(columns = ["rr_lag_1"], inplace = True)
        df_temp.drop(columns = ["casos_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_1"], inplace = True)
        df_temp.drop(columns = ["rr_lag_2"], inplace = True)
        df_temp.drop(columns = ["casos_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_2"], inplace = True)
        df_temp.drop(columns = ["rr_lag_3"], inplace = True)
        df_temp.drop(columns = ["casos_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_3"], inplace = True)
        df_temp.drop(columns = ["rr_lag_4"], inplace = True)
        df_temp.drop(columns = ["casos_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_4"], inplace = True)
        df_temp.drop(columns = ["rr_lag_5"], inplace = True)
        df_temp.drop(columns = ["casos_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_5"], inplace = True)
        df_temp.drop(columns = ["rr_lag_6"], inplace = True)
        df_temp.drop(columns = ["casos_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_6"], inplace = True)
        df_temp.drop(columns = ["rr_lag_7"], inplace = True)
        df_temp.drop(columns = ["casos_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_7"], inplace = True)
        df_temp.drop(columns = ["rr_lag_8"], inplace = True)
        df_temp.drop(columns = ["casos_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_8"], inplace = True)

        df_temp = pd.get_dummies(df_temp, columns = ["urb"])

        train = df_temp[df_temp['week_canton'] < '2024-1-101']
        val = df_temp[(df_temp['week_canton'] >= '2024-1-101') & (df_temp['week_canton'] < '2025-1-101')]
        test = df_temp[df_temp['week_canton'] >= '2025-1-101']

        X_train = train.drop(columns = ["clasi_rr_no_0", "week_canton"])
        y_train = train["clasi_rr_no_0"]

        X_val = val.drop(columns = ["clasi_rr_no_0", "week_canton"])
        y_val= val["clasi_rr_no_0"]

        X_test= test.drop(columns = ["clasi_rr_no_0", "week_canton"])
        y_test = test["clasi_rr_no_0"]

        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)

        split_indices = np.zeros(len(X_combined))
        split_indices[:len(X_train)] = -1
        split_indices[len(X_train):] = 0

        pds = PredefinedSplit(test_fold=split_indices)

        scaler = StandardScaler()  

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)  
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        X_test_for_reg = pd.concat([X_val, X_test], axis=0)
        y_test_for_reg = pd.concat([y_val, y_test], axis=0)
        val_test = pd.concat([val, test], axis=0)

        X_test_for_reg_scaled = pd.DataFrame(scaler.transform(X_test_for_reg), columns=X_test_for_reg.columns)

        return X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds

    def serie_temp_canton(self, var, model, momento, canton):

        results = pd.read_csv(f'../../data/model_results/{momento}/results_{var}_{model}_{momento}_{canton}.csv')

        rmse = root_mean_squared_error(results["actual"], results["pred"])
        mse = rmse ** 2
        mae = mean_absolute_error(results["actual"], results["pred"])

        fig, ax = plt.subplots()
        ax.plot(results['week_canton'], results["actual"], label='Reales', marker='o')
        ax.plot(results['week_canton'], results["pred"], label='Predichos', marker='o')
        self.ticks_years_top(ax, 20)
        ax.set_xlabel('Week')
        ax.set_ylabel(f'{var}')

        ax.text(0.9, 0.75, f'MSE: {mse:.2f}', transform=ax.transAxes, ha='center', fontsize=12)
        ax.text(0.9, 0.68, f'RMSE: {rmse:.2f}', transform=ax.transAxes, ha='center', fontsize=12)
        ax.text(0.9, 0.55, f'MAE: {mae:.2f}', transform=ax.transAxes, ha='center', fontsize=12)
        ax.set_title(f"{model} for {canton}")
        plt.show()
    
    def importance_classic(self, var, X_train, X_train_scaled, model):
        coef_df = pd.DataFrame({"Feature": X_train.columns, "Coefficient": model.coef_})
        sd = pd.DataFrame({"Feature": X_train.columns, "Std": X_train_scaled.std()})
        coef_df = coef_df.merge(sd, on = "Feature")
        coef_df["imp_reg"] = coef_df["Coefficient"] * coef_df["Std"]
        coef_reg_sorted = coef_df.sort_values(by = "imp_reg", ascending=False).reset_index()

        print(coef_reg_sorted)

        fig, ax = plt.subplots(figsize=(10, 8))
        coef_reg_sorted = coef_df.sort_values(by = "imp_reg", ascending=True)
        ax.barh(coef_reg_sorted["Feature"].tail(10), coef_reg_sorted["imp_reg"].tail(10))
        ax.bar_label(ax.containers[0], fmt='%.2f')
        ax.set_xlabel("Coefficient")
        fig.suptitle(f"Linear regression variable importance for {var}")

        plt.show()

        coef_reg_sorted = coef_df.sort_values(by = "imp_reg", ascending=False).reset_index()

        return coef_reg_sorted

    def importance_ml(self, var, X_train, X_test_for_reg, y_test_for_reg, model, model_type):
        try:
            effective_model = model.best_estimator_
        except:
            effective_model = model

        r = permutation_importance(effective_model, X_test_for_reg, y_test_for_reg, n_repeats=30, random_state=0)

        imp_df = pd.DataFrame({"Feature": X_train.columns[r.importances_mean.argsort()[::-1]], f"imp_{model_type}": r.importances_mean[r.importances_mean.argsort()[::-1]]})

        print(imp_df)

        fig, ax = plt.subplots(figsize=(10, 8))
        imp_df_sorted = imp_df.sort_values(by = f"imp_{model_type}", ascending=True)
        ax.barh(imp_df_sorted["Feature"].tail(10), imp_df_sorted[f"imp_{model_type}"].tail(10))
        ax.bar_label(ax.containers[0], fmt='%.2f')
        ax.set_xlabel("Importance")
        fig.suptitle(f"{model_type} permutation importance for {var}")
        plt.show()

        return imp_df

    def rfecv_selection(self, var, model_type, model, X_train, y_train, X_test_for_reg, y_test_for_reg, canton, momento):

        rfecv = RFECV(
            estimator=model,
            step=1,
            cv=5,
            min_features_to_select=5,
            n_jobs=-1
        )

        rfecv.fit(X_train, y_train)

        print(f'Optimal number of features for {var} and {model_type}: {rfecv.n_features_}')
        selected_rfe = [f for f, s in zip(X_train.columns.tolist(), rfecv.support_) if s]
        print(f'Selected features: {selected_rfe}')

        fig, ax = plt.subplots(figsize=(9, 4))
        n_features_range = range(rfecv.min_features_to_select,
                                rfecv.min_features_to_select + len(rfecv.cv_results_['mean_test_score']))
        ax.plot(n_features_range, rfecv.cv_results_['mean_test_score'], linewidth=2, marker='o', markersize=4)
        ax.fill_between(
            n_features_range,
            rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'],
            rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'],
            alpha=0.2
        )
        ax.axvline(rfecv.n_features_, linestyle='--', linewidth=2,
                label=f'Optimal: {rfecv.n_features_} features')
        ax.set_title(f'RFECV: Metric (AUC or) vs Number of Features {var} and {model_type}', pad=10)
        ax.set_xlabel('Number of Features Selected')
        ax.set_ylabel('Cross-Validated metric (AUC or)')
        ax.legend()
        plt.tight_layout()
        plt.show()

        feature_names = X_train.columns[rfecv.support_]

        y_pred = rfecv.estimator_.predict(X_test_for_reg[feature_names])

        # if canton != "full":
        #     self.serie_temp_canton(var, model_type, momento, canton)
        # else:
        #     mae = mean_absolute_error(y_test_for_reg, y_pred)
        #     rmse = root_mean_squared_error(y_test_for_reg, y_pred)

        # print(f"""MAE reg: {round(mae, 3)}
        # RMSE reg: {round(rmse, 3)}
        #         """)
        # if var == "rr":
        #     nrmse = rmse / np.mean(y_test_for_reg)
        #     print(f"nrmse: {round(nrmse, 2)}")

        if model_type == "Linear Regression":
            importance_scores  = rfecv.estimator_.coef_ * X_train[feature_names].std()
        else:
            importance_scores = permutation_importance(rfecv.estimator_, X_test_for_reg[feature_names], y_test_for_reg, n_repeats=30, random_state=0).importances_mean

        df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})
        df_importance = df_importance.sort_values(by='Importance', ascending=True)

        df_importance.plot(kind='barh', x='Feature', y='Importance', legend=False)
        plt.title(f"Feature Importance Selected by RFECV for {model_type} model of {var}")
        plt.show()

    def linear_reg(self, var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds):

        reg = LinearRegression().fit(X_combined_scaled, y_combined)

        y_pred_reg = reg.predict(X_test_scaled)
        y_pred_train = reg.predict(X_combined_scaled)

        results_reg = pd.DataFrame({
            'actual': y_test,   
            'pred': y_pred_reg, 
            "week_canton": test["week_canton"].values 
        })

        results_reg.to_csv(f'../../data/model_results/{momento}/results_{var}_reg_{momento}_{canton}.csv')
        
        mae_reg = mean_absolute_error(y_test, y_pred_reg)
        rmse_reg = root_mean_squared_error(y_test, y_pred_reg)
        rmse_reg_train = root_mean_squared_error(y_combined, y_pred_train)

        if canton != "full":
            self.serie_temp_canton(var, "reg", momento, canton)
        else:
            print(f"""MAE reg: {round(mae_reg, 3)}
            RMSE reg: {round(rmse_reg, 3)}
                """)
            
        nrmse_reg = rmse_reg / np.mean(y_test)
        print(f"nrmse: {round(nrmse_reg, 2)}")

        print(f"""
            RMSE train reg: {round(rmse_reg_train, 3)}
            MSE train reg: {round(rmse_reg_train ** 2, 3)}
                """)
            
        fig, ax = plt.subplots()
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.scatter(y_test, y_pred_reg)
        fig.suptitle(f"Linear regression real vs predicted for {var}")
        plt.show()

        # coef_reg_sorted = self.importance_classic(var = var, X_train = X_train, X_train_scaled = X_train_scaled, model = reg)

        # self.rfecv_selection(var = var, model_type = "Linear Regression", model = reg, X_train = X_train_scaled, y_train = y_train, _test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, canton = canton, momento = momento)

        return reg

    def rf_reg(self, var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds):
        rf = RandomForestRegressor(oob_score=True, random_state=42)

        # param_grid_rf = {
        #     'max_depth': [1, 2, 3],
        #     "max_features": ["sqrt", "log2"],
        #     'min_samples_split': [20, 30, 40],
        #     'ccp_alpha': [0], 
        #     "criterion": ["squared_error"]
        # }

        param_grid_rf = {
            'max_depth': [5, 6, 7, 8],
            'min_samples_split': [10, 100, 500],
            'ccp_alpha': [0, 1 /  10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10], 
            "criterion": ["squared_error", "absolute_error"]
        }

        grid_rf = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid_rf, 
            cv=pds, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        grid_rf.fit(X_combined, y_combined)

        y_pred_rf = grid_rf.predict(X_test)
        y_pred_train = grid_rf.predict(X_combined)

        results_rf = pd.DataFrame({
            'actual': y_test,   
            'pred': y_pred_rf, 
            "week_canton": test["week_canton"].values
        })

        results_rf.to_csv(f'../../data/model_results/{momento}/results_{var}_rf_{momento}_{canton}.csv')

        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        rmse_rf = root_mean_squared_error(y_test, y_pred_rf)
        rmse_rf_train = root_mean_squared_error(y_combined, y_pred_train)

        if canton != "full":
            self.serie_temp_canton(var, "rf", momento, canton)
        else:
            print(f"""MAE rf: {round(mae_rf, 3)}
            RMSE rf: {round(rmse_rf, 3)}
                """)
       
        nrmse_rf = rmse_rf / np.mean(y_test)
        print(f"nrmse: {round(nrmse_rf, 2)}")

        print(f"""
            RMSE train rf: {round(rmse_rf_train, 3)}
            MSE train rf: {round(rmse_rf_train ** 2, 3)}
                """)

        fig, ax = plt.subplots()
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.scatter(y_test, y_pred_rf)
        fig.suptitle(f"RF real vs predicted for {var}")
        plt.show()

        # imp_rf = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = grid_rf, model_type = "rf")

        # self.rfecv_selection(var = var, model_type = "rf", model = grid_rf.best_estimator_, X_train = X_train, y_train = y_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, canton = canton, momento = momento)

        return grid_rf

    def xgb_reg(self, var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds):

        xgb = XGBRegressor(random_state=42)

        # param_grid_xgb = {
        #     'max_depth': [5],
        #     'learning_rate': [0.1],
        #     'n_estimator': [50], 
        #     "criterion": ["friednman_mse"]
        # }

        param_grid_xgb = {
            'max_depth': [5, 6, 7, 8],
            'learning_rate': [0.1, 0.3, 0.05],
            'n_estimator': [50, 100, 150], 
            "criterion": ["friednman_mse", "squared_error"]
        }

        grid_xgb = GridSearchCV(
            estimator=xgb, 
            param_grid=param_grid_xgb, 
            cv=pds, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )

        grid_xgb.fit(X_combined, y_combined)

        y_pred_xgb = grid_xgb.predict(X_test)
        y_pred_train = grid_xgb.predict(X_combined)

        results_xgb = pd.DataFrame({
            'actual': y_test,   
            'pred': y_pred_xgb, 
            "week_canton": test["week_canton"].values
        })

        results_xgb.to_csv(f'../../data/model_results/{momento}/results_{var}_xgb_{momento}_{canton}.csv')

        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        rmse_xgb = root_mean_squared_error(y_test, y_pred_xgb)
        rmse_rf_train = root_mean_squared_error(y_combined, y_pred_train)

        if canton != "full":
            self.serie_temp_canton(var, "xgb", momento, canton)
        else:
            print(f"""MAE xgb: {round(mae_xgb, 3)}
            RMSE xgb: {round(rmse_xgb, 3)}
                """)

        nrmse_rf = rmse_xgb / np.mean(y_test)
        print(f"nrmse: {round(nrmse_rf, 2)}")

        print(f"""
            RMSE train rf: {round(rmse_rf_train, 3)}
            MSE train xgb: {round(rmse_rf_train ** 2, 3)}
                """)

        fig, ax = plt.subplots()
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.scatter(y_test, y_pred_xgb)
        fig.suptitle(f"XGB real vs predicted for {var}")
        plt.show()

        # imp_xgb = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = grid_xgb, model_type = "xgb")

        # self.rfecv_selection(var = var, model_type = "XGB Regression", model = grid_xgb.best_estimator_, X_train = X_train, y_train = y_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, canton = canton, momento = momento)

        return grid_xgb

    def roc_auc_metrics_log(self, var, class_of_interest, y_score_reg, label_binarizer, y_onehot_test):
        class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

        display = RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score_reg[:, class_id],
            name=f"{class_of_interest} contra el resto",
            curve_kwargs=dict(color="darkorange"),
            plot_chance_level=True,
            despine=True,
        )
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Logistic: One-vs-Rest ROC curves for {var}:\n{class_of_interest} vs el resto)",
        )

    def roc_auc_metrics_rf(self, var, class_of_interest, y_score_rf, label_binarizer, y_onehot_test):
        class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

        display = RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score_rf[:, class_id],
            name=f"{class_of_interest} contra el resto",
            curve_kwargs=dict(color="darkorange"),
            plot_chance_level=True,
            despine=True,
        )
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"RF: One-vs-Rest ROC curves for {var}:\n{class_of_interest} vs el resto",
        )

    def roc_auc_metrics_xgb(self, var, class_of_interest, y_score_xgb, y_true, class_index):

        if len(np.unique(y_true)) < 2:
            print(f"Skipping ROC for {class_of_interest}: only one class present")
            return

        y_true_binary = (y_true == class_index).astype(int)

        if y_score_xgb.ndim == 1:
            y_score_class = y_score_xgb
        else:
            y_score_class = y_score_xgb[:, class_index]

        display = RocCurveDisplay.from_predictions(
            y_true_binary,
            y_score_class,
            name=f"{class_of_interest} vs resto",
            curve_kwargs=dict(color="darkorange"),
            plot_chance_level=True,
            despine=True,
        )

        display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"XGB: ROC One-vs-Rest for {var}\n{class_of_interest} vs resto",
        )

        plt.show()

    def log_reg(self, var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds):
        reg = LogisticRegression()

        reg.fit(X_combined_scaled, y_combined)

        y_pred_reg = reg.predict(X_test_scaled)

        results_reg = pd.DataFrame({
            'actual': y_test,   
            'pred': y_pred_reg,
            "week_canton": val_test["week_canton"].values # FIX
        })

        results_reg.to_csv(f'../../data/model_results/{momento}/results_{var}_reg_{momento}_{canton}.csv')

        print(classification_report(y_test, y_pred_reg))

        label_binarizer = LabelBinarizer().fit(y_combined)
        y_onehot_test = label_binarizer.transform(y_test)

        y_score_reg = reg.predict_proba(X_combined_scaled)

        known_classes = set(label_binarizer.classes_)

        classes = ["Alto", "Bajo","Medio"]

        for target in classes:
            if target not in known_classes:
                print(f"Skipping roc_auc_metrics_log for '{target}': not in training set")
                continue

            self.roc_auc_metrics_log(
                var, target,
                y_score_reg, label_binarizer, y_onehot_test
            )

        cnf_matrix_reg = confusion_matrix(y_test, y_pred_reg)
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_reg, display_labels = ["Alto", "Bajo", "Medio"])
        disp.plot()
        disp.ax_.set_title(f"Logistic regression confusion matrix for {var}")

        # self.rfecv_selection(var = var, model_type = "Logistic Regression", model = reg, X_train = X_train_scaled, y_train = y_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, canton = canton, momento = momento)

        return reg

    def rf_classi(self, var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds):
        rf = RandomForestClassifier(oob_score=True, random_state=42)

        param_grid_rf = {
            'max_depth': [5],
            'min_samples_split': [10],
            'ccp_alpha': [0], 
            "criterion": ["gini"]
        }

        # param_grid_rf = {
        #     'max_depth': [5, 6, 7, 8],
        #     'min_samples_split': [10, 100, 500],
        #     'ccp_alpha': [0, 1 /  10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10], 
        #     "criterion": ["gini", "entropy"]
        # }

        grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv = pds, n_jobs=-1, verbose=10)
        grid_rf.fit(X_combined, y_combined)

        y_pred_rf = grid_rf.predict(X_combined)

        results_rf = pd.DataFrame({
            'actual': y_test,   
            'pred': y_pred_rf,
            "week_canton": test["week_canton"].values
        })

        results_rf.to_csv(f'../../data/model_results/{momento}/results_{var}_rf_{momento}_{canton}.csv')

        y_score_rf = grid_rf.predict_proba(X_test)

        label_binarizer = LabelBinarizer().fit(y_combined)
        y_onehot_test = label_binarizer.transform(y_test)

        print(classification_report(y_test, y_pred_rf))

        known_classes = set(label_binarizer.classes_)

        classes = ["Alto", "Bajo","Medio"]

        for target in classes:
            if target not in known_classes:
                print(f"Skipping roc_auc_metrics_rf for '{target}': not in training set")
                continue

            self.roc_auc_metrics_rf(
                var, target,
                y_score_rf, label_binarizer, y_onehot_test
            )

        cnf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_rf)
        disp.plot()
        disp.ax_.set_title(f"RF confusion matrix for {var}")

        # imp_rf = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = grid_rf, model_type = "rf")

        # self.rfecv_selection(var = var, model_type = "RF Classifier", model = grid_rf.best_estimator_, X_train = X_train, y_train = y_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, canton = canton, momento = momento)

        return grid_rf

    def xgb_classi(self, var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds):

        classes_in_train = set(y_combined)
        mask = y_test.isin(classes_in_train)

        X_test = X_test.loc[mask]
        y_test = y_test[mask]

        le = LabelEncoder()
        y_combined_encoded = le.fit_transform(y_combined)
        y_test_encoded = le.transform(y_test)

        xgb = XGBClassifier(random_state=42)

        param_grid_xgb = {
            'max_depth': [5],
            'learning_rate': [0.3],
            'n_estimators': [50],
            "reg_lambda": [0]
        }

        # param_grid_xgb = {
        #     'max_depth': [5, 6, 7, 8],
        #     'learning_rate': [0.3, 0.1, 0.05],
        #     'n_estimators': [50, 100, 150], 
        #     "reg_lambda": [0, 1 /  10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10],
        # }

        grid_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv = pds, n_jobs=-1, verbose=0)
        grid_xgb.fit(X_combined, y_combined_encoded)

        y_pred_xgb = grid_xgb.predict(X_test)

        test_filtered = test.loc[mask]

        results_xgb = pd.DataFrame({
            'actual': y_test_encoded,   
            'pred': y_pred_xgb,
            "week_canton": test_filtered["week_canton"].values
        })

        results_xgb.to_csv(f'../../data/model_results/{momento}/results_{var}_xgb_{momento}_{canton}.csv')

        y_score_xgb = grid_xgb.predict_proba(X_test)

        n_classes = len(le.classes_)

        if n_classes == 2:
            y_true_global = y_test_encoded
            y_score_global = y_score_xgb[:, 1]
        else:
            y_true_global = label_binarize(y_test_encoded, classes=range(n_classes))
            y_score_global = y_score_xgb

        print(classification_report(y_test_encoded, y_pred_xgb))

        for i, class_name in enumerate(le.classes_):

            if n_classes == 2:
                self.roc_auc_metrics_xgb(
                    var,
                    class_name,
                    y_score_global,
                    y_true_global,
                    class_index=i
                )
            else:
                self.roc_auc_metrics_xgb(
                    var,
                    class_name,
                    y_score_global[:, i],
                    y_true_global[:, i],
                    class_index=i
                )

        cnf_matrix_xgb = confusion_matrix(y_test_encoded, y_pred_xgb)
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_xgb)
        disp.plot()
        disp.ax_.set_title(f"XGB confusion matrix for {var}")

        # imp_xgb = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_encoded, model = grid_xgb, model_type = "xgb")

        # self.rfecv_selection(var = var, model_type = "XGB Classifier", model = grid_xgb.best_estimator_, X_train = X_train,y_train = y_train_encoded, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_encoded,   canton = canton, momento = momento)

        return grid_xgb

    def hybrid(self, var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds):
        # FIX

        y_train_bin = np.where(y_train > 0, 1, 0)

        if classi_type == "reg":
            reg = LogisticRegression()

            reg.fit(X_train_scaled, y_train_bin)

            y_pred_classi = reg.predict(X_test_for_reg_scaled)

        elif classi_type == "rf":
            rf = RandomForestClassifier(oob_score=True)

            param_grid_rf = {
                'max_depth': [5],
                'min_samples_split': [10],
                'ccp_alpha': [0], 
                "criterion": ["gini"]
            }

            # param_grid_rf = {
            #     'max_depth': [5, 6, 7, 8],
            #     'min_samples_split': [10, 100, 500],
            #     'ccp_alpha': [0, 1 /  10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10], 
            #     "criterion": ["gini", "entropy"]
            # }

            grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv = 5, n_jobs=-1, verbose=10)
            grid_rf.fit(X_train, y_train_bin)

            y_pred_classi = grid_rf.predict(X_test_for_reg)

        elif classi_type == "xgb":
            classes_in_train = set(y_train)
            mask = y_test_for_reg.isin(classes_in_train)

            X_test_for_reg = X_test_for_reg.loc[mask]
            y_test_for_reg = y_test_for_reg[mask]

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_bin)
            y_test_encoded = le.transform(y_test_for_reg)

            xgb = XGBClassifier()

            param_grid_xgb = {
                'max_depth': [5],
                'learning_rate': [0.3],
                'n_estimators': [50],
                "reg_lambda": [0]
            }

            # param_grid_xgb = {
            #     'max_depth': [5, 6, 7, 8],
            #     'learning_rate': [0.3, 0.1, 0.05],
            #     'n_estimators': [50, 100, 150], 
            #     "reg_lambda": [0, 1 /  10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10],
            # }

            grid_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv = 5, n_jobs=-1, verbose=0)
            grid_xgb.fit(X_train, y_train_encoded)

            y_pred_classi = grid_xgb.predict(X_test_for_reg)

        else: 
            print("Invalid classification model")

        if reg_type == "reg":
            X_train_scaled_1 = X_train_scaled[y_train > 0]
            y_train_1 = y_train[y_train > 0]

            reg_model = LinearRegression().fit(X_train_scaled_1, y_train_1)

            y_pred_reg = reg_model.predict(X_test_for_reg_scaled[y_pred_classi == 1])
        elif reg_type == "rf":
            X_train_1 = X_train[y_train > 0]
            y_train_1 = y_train[y_train > 0]

            rf = RandomForestRegressor(oob_score=True)

            param_grid_rf = {
                'max_depth': [5],
                'min_samples_split': [10],
                'ccp_alpha': [0], 
                "criterion": ["squared_error"]
            }

            # param_grid_rf = {
            #     'max_depth': [5, 6, 7, 8],
            #     'min_samples_split': [10, 100, 500],
            #     'ccp_alpha': [0, 1 /  10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10], 
            #     "criterion": ["squared_error", "absolute_error"]
            # }

            grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv = 5, n_jobs=-1, verbose=10)
            grid_rf.fit(X_train_1, y_train_1)

            X_test_for_reg_1 = X_test_for_reg[y_pred_classi == 1]

            y_pred_reg = grid_rf.predict(X_test_for_reg_1)

            reg_model = grid_rf.best_estimator_
        
        elif reg_type == "xgb":
            X_train_1 = X_train[y_train > 0]
            y_train_1 = y_train[y_train > 0]

            xgb = XGBRegressor()

            param_grid_xgb = {
                'max_depth': [5],
                'learning_rate': [0.1],
                'n_estimator': [50], 
                "criterion": ["friednman_mse"]
            }

            # param_grid_xgb = {
            #     'max_depth': [5, 6, 7, 8],
            #     'learning_rate': [0.1, 0.3, 0.05],
            #     'n_estimator': [50, 100, 150], 
            #     "criterion": ["friednman_mse", "squared_error"]
            # }

            grid_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv = 5, n_jobs=-1, verbose=0)
            
            grid_xgb.fit(X_train_1, y_train_1)

            X_test_for_reg_1 = X_test_for_reg[y_pred_classi == 1]

            y_pred_reg = grid_xgb.predict(X_test_for_reg[y_pred_classi == 1])

            reg_model = grid_xgb.best_estimator_

        else: 
            print("Invalid regression model")

        results = pd.DataFrame({
            'actual': y_test_for_reg[y_pred_classi == 1],   
            'pred': y_pred_reg,
            "week_canton": val_test["week_canton"].values
        })

        results.to_csv(f'../../data/model_results/{momento}/results_{var}_hybrid_{classi_type}_{reg_type}_{momento}_{canton}.csv')
        
        if reg_type == "reg":
            y_pred_train = reg_model.predict(X_train_scaled[y_pred_classi == 1])
        else:
            y_pred_train = reg_model.predict(X_train[y_pred_classi == 1])

        mae = mean_absolute_error(y_test_for_reg[y_pred_classi == 1], y_pred_reg)
        rmse = root_mean_squared_error(y_test_for_reg[y_pred_classi == 1], y_pred_reg)
        mse = rmse ** 2
        rmse_train = root_mean_squared_error(y_train[y_pred_classi == 1], y_pred_train)

        if canton != "full":
            self.serie_temp_canton(var, f"hybrid_{classi_type}_{reg_type}", momento, canton)
        else:
            print(f"""MAE: {round(mae, 3)}
            RMSE: {round(rmse, 3)}
                """)
       
        nrmse = rmse / np.mean(y_test_for_reg[y_pred_classi == 1])
        print(f"nrmse: {round(nrmse, 2)}")
        print(f"""
            MSE test: {round(mse, 3)}
            RMSE test: {round(rmse, 3)}
            RMSE train: {round(rmse_train, 3)}
            MSE train: {round(rmse_train ** 2, 3)}
                """)

        # fig, ax = plt.subplots()
        # ax.plot([min(y_test_for_reg), max(y_test_for_reg)], [min(y_test_for_reg), max(y_test_for_reg)], color='red', linestyle='--')
        # ax.scatter(y_test_for_reg, y_pred_reg)
        # fig.suptitle(f"hybrid_{classi_type}_{reg_type} real vs predicted for {var}")
        # plt.show()

        # if reg_type == "reg":
        #     imp = self.importance_classic(var = var, X_train = X_train, X_train_scaled = X_train_scaled, model = reg_model)
        # else: 
        #     imp = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = reg_model, model_type = f"hybrid_{classi_type}_{reg_type}")

        # if reg_type == "reg":
        #     self.rfecv_selection(var = var, model_type = f"hybrid_{classi_type}_{reg_type}", model = reg_model, X_train = X_train_scaled_1, y_train = y_train_1, X_test_for_reg = X_test_for_reg_scaled, y_test_for_reg = y_test_for_reg, canton = canton, momento = momento)
        # else:
        #     self.rfecv_selection(var = var, model_type = f"hybrid_{classi_type}_{reg_type}", model = reg_model, X_train = X_train_1, y_train = y_train_1, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, canton = canton, momento = momento)

        return reg_model

    def ticks_years_top(self, ax, n):
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        for i, tick in enumerate(ax.xaxis.get_major_ticks()):
            if i % n != 0:
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.gridline.set_visible(False) 

    def model_results(self, var, momento, canton = "full"):
        if var == "cases":

            X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds = self.partition_cases(momento)

            print("********** LINEAR REGRESSION **********")

            reg = self.linear_reg(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            print("********** RF **********")

            grid_rf = self.rf_reg(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            print("********** XGB **********")

            grid_xgb = self.xgb_reg(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            # coef_reg_sorted = self.importance_classic(var = var, X_train = X_train, X_train_scaled = X_train_scaled, model = reg)

        elif var == "rr":
            X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds = self.partition_rr(momento)

            print("********** LINEAR REGRESSION **********")

            reg = self.linear_reg(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            print("********** RF **********")

            grid_rf = self.rf_reg(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            print("********** XGB **********")

            grid_xgb = self.xgb_reg(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            # coef_reg_sorted = self.importance_classic(var = var, X_train = X_train, X_train_scaled = X_train_scaled, model = reg)
            
        elif var in ["classi_rr", "clasi_rr"]:
            X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds = self.partition_classi_rr(momento)

            print("********** LOGISTIC REGRESSION **********")

            reg = self.log_reg(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            print("********** RF **********")

            grid_rf = self.rf_classi(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            print("********** XGB **********")

            grid_xgb = self.xgb_classi(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            # coef_reg_sorted = self.importance_ml(var = var, X_train = X_train_scaled, X_test_for_reg = X_test_for_reg_scaled, y_test_for_reg = y_test_for_reg, model = reg, model_type = "logistic")

        elif var in ["classi_no_0", "clasi_no_0"]:
            X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds = self.partition_classi_rr_no_0(momento) 

            print("********** LOGISTIC REGRESSION **********")

            reg = self.log_reg(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            print("********** RF **********")

            grid_rf = self.rf_classi(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            print("********** XGB **********")

            grid_xgb = self.xgb_classi(var, momento, canton, X_combined, y_combined, X_test, y_test, X_combined_scaled, X_test_scaled, test, pds)

            # coef_reg_sorted = self.importance_ml(var = var, X_train = X_train_scaled, X_test_for_reg = X_test_for_reg_scaled, y_test_for_reg = y_test_for_reg, model = reg, model_type = "logistic")
            
        else:
            print("Invalid variable")

        # imp_rf_df = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = grid_rf, model_type = "rf")
        
        # if var in ["classi_rr", "clasi_rr", "classi_no_0", "clasi_no_0"]:
        #     le = LabelEncoder()
        #     y_test_for_reg_encoded = le.fit_transform(y_test_for_reg)
        #     imp_xgb_df = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg_encoded, model = grid_xgb, model_type = "xgb")
        # else:
        #     imp_xgb_df = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = grid_xgb, model_type = "xgb")

        # importances = pd.merge(coef_reg_sorted, imp_rf_df, on = "Feature")
        # importances = pd.merge(importances, imp_xgb_df, on = "Feature")

        # cols_to_drop = ["index", "Coefficient", "Std"]

        # if all(col in importances.columns for col in cols_to_drop):
        #     importances.drop(columns=cols_to_drop, inplace=True)

        # importances.to_excel(f'../../data/model_results/feature_importance/imp_{var}_{momento}_{canton}.xlsx')