import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, r2_score, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import joblib
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV

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

        scaler = StandardScaler()  

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)  
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        X_test_for_reg = pd.concat([X_val, X_test], axis=0)
        y_test_for_reg = pd.concat([y_val, y_test], axis=0)
        val_test = pd.concat([val, test], axis=0)

        X_test_for_reg_scaled = pd.DataFrame(scaler.transform(X_test_for_reg), columns=X_test_for_reg.columns)

        return X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled

    def partition_rr(self, momento):
        df_temp = self.df.copy()

        df_temp.drop(columns=['clasi_rr'], inplace=True)
        df_temp.drop(columns = ["rr"], inplace = True)
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

        scaler = StandardScaler()  

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)  
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        X_test_for_reg = pd.concat([X_val, X_test], axis=0)
        y_test_for_reg = pd.concat([y_val, y_test], axis=0)
        val_test = pd.concat([val, test], axis=0)

        X_test_for_reg_scaled = pd.DataFrame(scaler.transform(X_test_for_reg), columns=X_test_for_reg.columns)

        return X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled

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

        scaler = StandardScaler()  

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)  
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        X_test_for_reg = pd.concat([X_val, X_test], axis=0)
        y_test_for_reg = pd.concat([y_val, y_test], axis=0)
        val_test = pd.concat([val, test], axis=0)

        X_test_for_reg_scaled = pd.DataFrame(scaler.transform(X_test_for_reg), columns=X_test_for_reg.columns)

        return X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled

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

        scaler = StandardScaler()  

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)  
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        X_test_for_reg = pd.concat([X_val, X_test], axis=0)
        y_test_for_reg = pd.concat([y_val, y_test], axis=0)
        val_test = pd.concat([val, test], axis=0)

        X_test_for_reg_scaled = pd.DataFrame(scaler.transform(X_test_for_reg), columns=X_test_for_reg.columns)

        return X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled
    
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
        fig.suptitle(f"Logistic or linear regression variable importance for {var}")

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

    def rfecv_selection(self, var, model_type, model, X_train, y_train):

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

    def linear_reg(self, var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled):

        reg = LinearRegression().fit(X_train_scaled, y_train)

        y_pred_reg = reg.predict(X_test_for_reg_scaled)

        results_reg = pd.DataFrame({
            'actual': y_test_for_reg,   
            'pred': y_pred_reg, 
            "week_canton": val_test["week_canton"].values
        })

        results_reg.to_csv(f'../../data/model_results/{momento}/results_{var}_reg_{momento}.csv')

        r2_reg = r2_score(y_test_for_reg, y_pred_reg)
        mae_reg = mean_absolute_error(y_test_for_reg, y_pred_reg)
        rmse_reg = root_mean_squared_error(y_test_for_reg, y_pred_reg)

        print(f"""R2 reg: {round(r2_reg, 3)}
        MAE reg: {round(mae_reg, 3)}
        RMSE reg: {round(rmse_reg, 3)}
            """)

        fig, ax = plt.subplots()
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.scatter(y_test_for_reg, y_pred_reg)
        fig.suptitle(f"Linear regression real vs predicted for {var}")
        plt.show()

        coef_reg_sorted = self.importance_classic(var = var, X_train = X_train, X_train_scaled = X_train_scaled, model = reg)

        self.rfecv_selection(var = var, model_type = "Linear Regression", model = reg, X_train = X_train_scaled, y_train = y_train)

        return reg

    def rf_reg(self, var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled):
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
        grid_rf.fit(X_train, y_train)

        y_pred_rf = grid_rf.predict(X_test_for_reg)

        results_rf = pd.DataFrame({
            'actual': y_test_for_reg,   
            'pred': y_pred_rf, 
            "week_canton": val_test["week_canton"].values
        })

        results_rf.to_csv(f'../../data/model_results/{momento}/results_{var}_rf_{momento}.csv')

        mae_rf = mean_absolute_error(y_test_for_reg, y_pred_rf)
        rmse_rf = root_mean_squared_error(y_test_for_reg, y_pred_rf)

        print(f"""MAE rf: {round(mae_rf, 3)}
        RMSE rf: {round(rmse_rf, 3)}
            """)

        fig, ax = plt.subplots()
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.scatter(y_test_for_reg, y_pred_rf)
        fig.suptitle(f"RF real vs predicted for {var}")
        plt.show()

        imp_rf = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = grid_rf, model_type = "rf")

        return grid_rf

    def xgb_reg(self, var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled):

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
        grid_xgb.fit(X_train, y_train)

        y_pred_xgb = grid_xgb.predict(X_test_for_reg)

        results_xgb = pd.DataFrame({
            'actual': y_test_for_reg,   
            'pred': y_pred_xgb, 
            "week_canton": val_test["week_canton"].values
        })

        results_xgb.to_csv(f'../../data/model_results/{momento}/results_{var}_xgb_{momento}.csv')

        mae_xgb = mean_absolute_error(y_test_for_reg, y_pred_xgb)
        rmse_xgb = root_mean_squared_error(y_test_for_reg, y_pred_xgb)

        print(f"""MAE xgb: {round(mae_xgb, 3)}
        RMSE xgb: {round(rmse_xgb, 3)}
            """)

        fig, ax = plt.subplots()
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.scatter(y_test_for_reg, y_pred_xgb)
        fig.suptitle(f"XGB real vs predicted for {var}")
        plt.show()

        imp_xgb = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = grid_xgb, model_type = "xgb")

        return grid_xgb

    def roc_auc_metrics_log(self, var, class_of_interest, other_classes, y_score_reg, label_binarizer, y_onehot_test):
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
            title=f"Logistic: One-vs-Rest ROC curves for {var}:\n{class_of_interest} vs ({other_classes[0]} and {other_classes[1]})",
        )

    def roc_auc_metrics_rf(self, var, class_of_interest, other_classes, y_score_rf, label_binarizer, y_onehot_test):
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
            title=f"RF: One-vs-Rest ROC curves for {var}:\n{class_of_interest} vs ({other_classes[0]} and {other_classes[1]})",
        )

    def roc_auc_metrics_xgb(self, var, class_of_interest, other_classes, y_score_xgb, label_binarizer, y_onehot_test):
        classes = ["Alto", "Bajo", "Medio"]

        class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

        display = RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score_xgb[:, class_id],
            name=f"{classes[class_of_interest]} contra el resto",
            curve_kwargs=dict(color="darkorange"),
            plot_chance_level=True,
            despine=True,
        )
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"XGB: One-vs-Rest ROC curves for {var}:\n{classes[class_of_interest]} vs ({classes[other_classes[0]]} and {classes[other_classes[1]]})",
        )

    def log_reg(self, var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled):
        reg = LogisticRegression()

        reg.fit(X_train_scaled, y_train)

        y_pred_reg = reg.predict(X_test_for_reg_scaled)

        results_reg = pd.DataFrame({
            'actual': y_test_for_reg,   
            'pred': y_pred_reg,
            "week_canton": val_test["week_canton"].values
        })

        results_reg.to_csv(f'../../data/model_results/{momento}/results_{var}_reg_{momento}.csv')

        print(classification_report(y_test_for_reg, y_pred_reg))

        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_test = label_binarizer.transform(y_test_for_reg)

        y_score_reg = reg.predict_proba(X_test_for_reg)

        self.roc_auc_metrics_log(var, "Alto", ["Bajo", "Medio"], y_score_reg, label_binarizer, y_onehot_test)

        self.roc_auc_metrics_log(var, "Bajo", ["Alto", "Medio"], y_score_reg, label_binarizer, y_onehot_test)

        self.roc_auc_metrics_log(var, "Medio", ["Alto", "Bajo"], y_score_reg, label_binarizer, y_onehot_test)

        cnf_matrix_reg = confusion_matrix(y_test_for_reg, y_pred_reg)
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_reg, display_labels = ["Alto", "Bajo", "Medio"])
        disp.plot()
        disp.ax_.set_title(f"Logistic regression confusion matrix for {var}")

        self.rfecv_selection(var = var, model_type = "Logistic Regression", model = reg, X_train = X_train_scaled, y_train = y_train)

        return reg

    def rf_classi(self, var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled):
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
        grid_rf.fit(X_train, y_train)

        y_pred_rf = grid_rf.predict(X_test_for_reg)

        results_rf = pd.DataFrame({
            'actual': y_test_for_reg,   
            'pred': y_pred_rf,
            "week_canton": val_test["week_canton"].values
        })

        results_rf.to_csv(f'../../data/model_results/{momento}/results_{var}_rf_{momento}.csv')

        y_score_rf = grid_rf.predict_proba(X_test_for_reg)

        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_test = label_binarizer.transform(y_test_for_reg)

        print(classification_report(y_test_for_reg, y_pred_rf))

        self.roc_auc_metrics_rf(var, "Alto", ["Bajo", "Medio"], y_score_rf, label_binarizer, y_onehot_test)

        self.roc_auc_metrics_rf(var, "Bajo", ["Alto", "Medio"], y_score_rf, label_binarizer, y_onehot_test)

        self.roc_auc_metrics_rf(var,"Medio", ["Bajo", "Alto"], y_score_rf, label_binarizer, y_onehot_test)

        cnf_matrix_rf = confusion_matrix(y_test_for_reg, y_pred_rf)
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_rf)
        disp.plot()
        disp.ax_.set_title(f"RF confusion matrix for {var}")

        imp_rf = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = grid_rf, model_type = "rf")

        return grid_rf

    def xgb_classi(self, var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled):

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test_for_reg)

        xgb = XGBClassifier()

        param_grid_xgb = {
            'max_depth': [5],
            'learning_rate': [0.3],
            'n_estimator': [50], 
            "criterion": ["auc"],
            "reg_lambda": [0],
        }

        # param_grid_xgb = {
        #     'max_depth': [5, 6, 7, 8],
        #     'learning_rate': [0.3, 0.1, 0.05],
        #     'n_estimator': [50, 100, 150], 
        #     "criterion": ["auc", "logloss"],
        #     "reg_lambda": [0, 1 /  10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10],
        # }

        grid_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv = 5, n_jobs=-1, verbose=0)
        grid_xgb.fit(X_train, y_train_encoded)

        y_pred_xgb = grid_xgb.predict(X_test_for_reg)

        results_xgb = pd.DataFrame({
            'actual': y_test_encoded,   
            'pred': y_pred_xgb,
            "week_canton": val_test["week_canton"].values
        })

        results_xgb.to_csv(f'../../data/model_results/{momento}/results_{var}_xgb_{momento}.csv')

        y_score_xgb = grid_xgb.predict_proba(X_test_for_reg)

        label_binarizer = LabelBinarizer().fit(y_train_encoded)
        y_onehot_test = label_binarizer.transform(y_test_encoded)

        print(classification_report(y_test_encoded, y_pred_xgb))
        
        self.roc_auc_metrics_xgb(var, 0, [1, 2], y_score_xgb, label_binarizer, y_onehot_test)
        
        self. roc_auc_metrics_xgb(var, 1, [0, 2], y_score_xgb, label_binarizer, y_onehot_test)
    
        self.roc_auc_metrics_xgb(var, 2, [1, 0], y_score_xgb, label_binarizer, y_onehot_test)

        cnf_matrix_xgb = confusion_matrix(y_test_encoded, y_pred_xgb)
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_xgb)
        disp.plot()
        disp.ax_.set_title(f"XGB confusion matrix for {var}")

        imp_xgb = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = grid_xgb, model_type = "xgb")

        return grid_xgb

    def model_results(self, var, momento):
        if var == "cases":

            X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled = self.partition_cases(momento)

            print("********** LINEAR REGRESSION **********")

            reg = self.linear_reg(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)

            coef_reg_sorted = self.importance_classic(var = var, X_train = X_train, X_train_scaled = X_train_scaled, model = reg)

            print("********** RF **********")

            grid_rf = self.rf_reg(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)

            print("********** XGB **********")

            grid_xgb = self.xgb_reg(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)

        elif var == "rr":
            X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled = self.partition_rr(momento)

            print("********** LINEAR REGRESSION **********")

            reg = self.linear_reg(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)

            coef_reg_sorted = self.importance_classic(var = var, X_train = X_train, X_train_scaled = X_train_scaled, model = reg)

            print("********** RF **********")

            grid_rf = self.rf_reg(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)

            print("********** XGB **********")

            grid_xgb = self.xgb_reg(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)
            
        elif var in ["classi_rr", "clasi_rr"]:
            X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled = self.partition_classi_rr(momento)

            print("********** LOGISTIC REGRESSION **********")

            reg = self.log_reg(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)

            coef_reg_sorted = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = reg, model_type = "logistic")

            print("********** RF **********")

            grid_rf = self.rf_classi(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)

            print("********** XGB **********")

            grid_xgb = self.xgb_classi(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)

        elif var in ["classi_no_0", "clasi_no_0"]:
            X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled = self.partition_classi_rr_no_0(momento) 

            print("********** LOGISTIC REGRESSION **********")

            reg = self.log_reg(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)

            print("********** RF **********")

            grid_rf = self.rf_classi(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)

            print("********** XGB **********")

            grid_xgb = self.xgb_classi(var, momento, X_train, y_train, X_val, y_val, X_test, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_test_for_reg, y_test_for_reg, val_test, X_test_for_reg_scaled)
            
        else:
            print("Invalid variable")

        coef_reg_sorted = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = reg, model_type = "logistic")

        imp_rf_df = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = grid_rf, model_type = "rf")
        
        imp_xgb_df = self.importance_ml(var = var, X_train = X_train, X_test_for_reg = X_test_for_reg, y_test_for_reg = y_test_for_reg, model = grid_xgb, model_type = "xgb")

        importances = pd.merge(coef_reg_sorted, imp_rf_df, on = "Feature")
        importances = pd.merge(importances, imp_xgb_df, on = "Feature")

        importances.drop(columns = ["index", "Coefficient", "Std"], inplace=True)

        importances.to_excel(f'../../data/model_results/feature_importance/imp_{var}_{momento}.xlsx')