import pandas as pd
from sklearn.metrics import classification_report, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit, TunedThresholdClassifierCV
from xgboost import XGBClassifier
import joblib
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from arch.bootstrap import CircularBlockBootstrap
from mapie.regression import CrossConformalRegressor
from sklearn.base import clone


class Model:
    def __init__(self, df, canton):
        self.canton = canton
        self.df = df[df["week_canton"].str.contains(canton)]
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

    def partition(self):
        df_temp = self.df.copy()

        df_temp.drop(columns=['clasi_rr'], inplace=True)
        df_temp.drop(columns = ["casos"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0"], inplace = True)

        df_temp = self.drop_lags(i = 5, f = 9, df_temp = df_temp)
        for i in range(5, 9):
            df_temp.drop(columns = [f"rr_lag_{i}"], inplace = True)

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

        epsilon = 0.2 # This epsilon was found through trial and error, looking at histograms for each value

        X_train = train.drop(columns = ["rr", "week_canton"]).reset_index(drop=True)
        y_train = np.log(train["rr"] + epsilon).reset_index(drop=True)

        X_val= val.drop(columns = ["rr", "week_canton"]).reset_index(drop=True)
        y_val= np.log(val["rr"]  + epsilon).reset_index(drop=True)

        X_test = test.drop(columns = ["rr", "week_canton"]).reset_index(drop=True)
        y_test = np.log(test["rr"] + epsilon).reset_index(drop=True)

        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)

        split_indices = np.zeros(len(X_combined))
        split_indices[:len(X_train)] = -1
        split_indices[len(X_train):] = 0

        pds = PredefinedSplit(test_fold=split_indices)

        return X_combined, y_combined, X_train, X_test, y_test, test, pds, epsilon

    def serie_temp_canton(self, model_type, lower_bound, upper_bound):

        results = pd.read_csv(f'../../data/model_results/results_{model_type}_{self.canton}.csv')
        
        results['week_canton'] = results['week_canton'].str.extract(r'(\d{4}-\d+)').iloc[:, 0].tolist()

        fig, ax = plt.subplots()
        ax.plot(results['week_canton'], results["actual"], label='Real', marker='o', linestyle = "dashed", color = "blue")
        ax.plot(results['week_canton'], results["pred"], label='Predicted', marker='o', color = "green")
        ax.fill_between(x = results['week_canton'], label = "CI (95%)", y1 = lower_bound, y2 = upper_bound, alpha = 0.2, color = "green")
        ax.legend(loc = "upper right")
        self.ticks_years_top(ax, 20)
        ax.set_xlabel('Week')
        ax.set_ylabel(f'Relative risk')

        ax.set_title(f"{model_type} for {self.canton}")
        plt.show()
    
    def var_importance(self, X_test, y_test, model_type, repeats, var = "RR"):

        try:
            grid = joblib.load(f'../../models/saved_models/{model_type}_reg_{self.canton}.joblib')
        except FileNotFoundError: 
            grid = joblib.load(f'../../models/saved_models/{model_type}_{self.canton}.joblib')
        model = grid.best_estimator_

        r = permutation_importance(model, X_test, y_test, n_repeats=repeats, random_state=42, scoring='neg_mean_squared_error', n_jobs = -1)

        imp_df = pd.DataFrame({"Feature": X_test.columns, f"imp_{model_type}": r.importances_mean, f"std_{model_type}": r.importances_std}).sort_values(by=f"imp_{model_type}", ascending=False).reset_index(drop=True)

        print(imp_df)

        fig, ax = plt.subplots(figsize=(10, 8))
        imp_df_sorted = imp_df.head(10).sort_values(by=f"imp_{model_type}", ascending=True)
        ax.barh(imp_df_sorted["Feature"], imp_df_sorted[f"imp_{model_type}"])
        ax.bar_label(ax.containers[0], fmt='%.2f')
        ax.set_xlabel("Importance")
        fig.suptitle(f"{model_type} permutation importance for {var} in {self.canton}")
        plt.show()

        return imp_df

    def rfecv_selection(self, model_type, X_combined, y_combined, X_test, y_test, pds, repeats, var = "RR"):

        try:
            grid = joblib.load(f'../../models/saved_models/{model_type}_reg_{self.canton}.joblib')
        except FileNotFoundError: 
            grid = joblib.load(f'../../models/saved_models/{model_type}_{self.canton}.joblib')
        
        model = grid.best_estimator_

        rfecv = RFECV(
            estimator=model,
            step=1,
            cv=pds,
            min_features_to_select=5,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )

        rfecv.fit(X_combined, y_combined)

        selected_mask = rfecv.support_
        selected_features = X_combined.columns[selected_mask].tolist()

        print(f'Optimal number of features for {var} and {model_type}: {rfecv.n_features_}')
        print(f'Selected features: {selected_features}')

        if "n_features" in rfecv.cv_results_:
            n_features_range = rfecv.cv_results_["n_features"]
        else:
            n_features_range = range(
                rfecv.min_features_to_select,
                rfecv.min_features_to_select + len(rfecv.cv_results_["mean_test_score"])
            )

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(n_features_range, rfecv.cv_results_['mean_test_score'], linewidth=2, marker='o', markersize=4)
        ax.fill_between(
            n_features_range,
            rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'],
            rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'],
            alpha=0.2
        )
        ax.axvline(rfecv.n_features_, linestyle='--', linewidth=2,
                label=f'Optimal: {rfecv.n_features_} features')
        ax.set_title(f'RFECV score vs Number of Features {var} and {model_type}', pad=10)
        ax.set_xlabel('Number of Features Selected')
        ax.set_ylabel('Cross-Validated score')
        ax.legend()
        plt.tight_layout()
        plt.show()

        final_model = clone(model)
        final_model.fit(X_combined[selected_features], y_combined)

        final_model_pred = final_model.predict(X_test[selected_features])
        rmse = root_mean_squared_error(y_test, final_model_pred)

        nrmse = rmse / np.mean(y_test)

        print(f"New NRMSE with selected features: {nrmse:.4f}")
        
        importance_scores = permutation_importance(final_model, X_test[selected_features], y_test, n_repeats=repeats, random_state=42, scoring='neg_mean_squared_error', n_jobs = -1).importances_mean

        df_importance = pd.DataFrame({'Feature': selected_features, 'Importance': importance_scores})
        df_importance = df_importance.sort_values(by='Importance', ascending=True)

        df_importance.plot(kind='barh', x='Feature', y='Importance', legend=False,
        figsize=(8, max(4, 0.35 * len(df_importance))))
        plt.title(f"Feature Importance Selected by RFECV for {model_type} model of {var}")
        plt.show()

    def prediction_intervals(self, X_combined, y_combined, X_test, epsilon, model_type, pds):

        try:
            grid = joblib.load(f'../../models/saved_models/{model_type}_reg_{self.canton}.joblib')
        except: 
            grid = joblib.load(f'../../models/saved_models/{model_type}_{self.canton}.joblib')

        best_rf = grid.best_estimator_

        mapie_reg = CrossConformalRegressor(
             estimator=best_rf,
             confidence_level=0.95,
             cv = pds
         )

        mapie_reg.fit_conformalize(X_combined, y_combined)

        y_pred_rf_cp, y_interval = mapie_reg.predict_interval(X_test)

        lower_bound = np.exp(y_interval[:, 0, 0]) - epsilon
        upper_bound = np.exp(y_interval[:, 1, 0]) - epsilon

        y_pred_point = np.exp(y_pred_rf_cp) - epsilon

        return y_pred_point, lower_bound, upper_bound

    def rf_reg(self, X_combined, y_combined, X_test, y_test, test, pds, epsilon):
        rf = RandomForestRegressor(oob_score=True, random_state=42)

        param_grid_rf = {
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [10, 50, 75, 100, 150],
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

        y_pred_rf = np.exp(grid_rf.predict(X_test)) - epsilon
        y_test = np.exp(y_test) - epsilon

        results_rf = pd.DataFrame({
              'actual': y_test,   
              'pred': y_pred_rf, 
              "week_canton": test["week_canton"].values
        })

        results_rf.to_csv(f'../../data/model_results/results_rf_{self.canton}.csv')

        joblib.dump(grid_rf, f'../../models/saved_models/rf_{self.canton}.joblib')

        print(f"RF Model Ready for {self.canton}")

    def xgb_reg(self, X_combined, y_combined, X_test, y_test, test, pds, epsilon):

        xgb = XGBRegressor(random_state=42)

        param_grid_xgb = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.1, 0.3, 0.05, 0.01],
            'n_estimators': [10, 50, 75, 100, 150], 
            "eval_metric": ["rmse", "mae"],
            "reg_lambda": [0, 1 / 10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10]
        }
        grid_xgb = GridSearchCV(
            estimator=xgb, 
            param_grid=param_grid_xgb, 
            cv=pds, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )

        grid_xgb.fit(X_combined, y_combined)

        y_test = np.exp(y_test) - epsilon
        y_pred_xgb = np.exp(grid_xgb.predict(X_test)) - epsilon

        results_xgb = pd.DataFrame({
            'actual': y_test,   
            'pred': y_pred_xgb, 
            "week_canton": test["week_canton"].values
        })

        results_xgb.to_csv(f'../../data/model_results/results_xgb_{self.canton}.csv')

        joblib.dump(grid_xgb, f'../../models/saved_models/xgb_{self.canton}.joblib')

        print(f"XGB Model Ready for {self.canton}")

    def hybrid(self, model_type, X_combined, y_combined, X_train, X_test, y_test, test, pds, epsilon):

        X_combined = X_combined.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_combined = pd.Series(y_combined).reset_index(drop=True)
        y_test = pd.Series(y_test).reset_index(drop=True)
        test = test.reset_index(drop=True)

        y_combined_actual = np.exp(y_combined) - epsilon
        y_test_actual = np.exp(y_test) - epsilon 
        y_train_bin = pd.Series((y_combined_actual > 0).astype(int)).reset_index(drop=True) 
        y_test_bin = pd.Series((y_test_actual > 0).astype(int)).reset_index(drop=True) 
        if y_test_bin.nunique() < 2:
            print(f"WARNING: Test set only contains one label: {y_test_bin.unique()[0]}")

        if model_type == "hrf" and y_test_bin.nunique() >= 2:
            clf = RandomForestClassifier(oob_score=True, random_state=42)
            param_grid = {
                "max_depth": [3, 5, 7, 9],
                "min_samples_split": [10, 50, 75, 100, 150],
                "ccp_alpha": [0, 1 / 10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10],
                "criterion": ["gini", "entropy"]
            }
            grid_classi = GridSearchCV(clf, param_grid, cv=pds, n_jobs=-1, verbose=0, scoring="roc_auc")
            grid_classi.fit(X_combined, y_train_bin)
            
            best_clf = grid_classi.best_estimator_
            try:
                clf_tuned = TunedThresholdClassifierCV(
                    estimator=best_clf,
                    scoring="roc_auc",
                    cv=5,
                    n_jobs=-1,
                    random_state=42
                )
                clf_tuned.fit(X_combined, y_train_bin)
                y_pred_classi = clf_tuned.predict(X_test)
                grid_classi = clf_tuned  
            except ValueError as e:
                print(f"Threshold tuning failed: {e}")
                print("Using best classifier directly without threshold tuning.")
                y_pred_classi = best_clf.predict(X_test)
                grid_classi = best_clf  

            print(classification_report(y_test_bin, y_pred_classi))
            # RocCurveDisplay.from_predictions(y_test_bin, clf.predict_proba(X_test)[:, 1], plot_chance_level= True)

        elif model_type == "hxgb" and y_test_bin.nunique() >= 2:
            clf = XGBClassifier(random_state=42)
            param_grid = {
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.1, 0.3, 0.05, 0.01],
                "n_estimators": [10, 50, 75, 100, 150],
                "reg_lambda": [0, 1 / 10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10],
                "eval_metric": ["logloss"]
            }
            grid_classi = GridSearchCV(clf, param_grid, cv=pds, n_jobs=-1, scoring="roc_auc")
            grid_classi.fit(X_combined, y_train_bin)
            best_clf = grid_classi.best_estimator_

            try:
                clf_tuned = TunedThresholdClassifierCV(
                    estimator=best_clf,
                    scoring="roc_auc",
                    cv=5,
                    n_jobs=-1,
                    random_state=42
                )
                clf_tuned.fit(X_combined, y_train_bin)
                y_pred_classi = clf_tuned.predict(X_test)
                grid_classi = clf_tuned  
            except ValueError as e:
                print(f"Threshold tuning failed: {e}")
                print("Using best classifier directly without threshold tuning.")
                y_pred_classi = best_clf.predict(X_test)
                grid_classi = best_clf  
            
            print(classification_report(y_test_bin, y_pred_classi))
            # RocCurveDisplay.from_predictions(y_test_bin, clf.predict_proba(X_test)[:, 1], plot_chance_level= True)
        elif y_test_bin.nunique() < 2:
            print("Only one class present in test set for classification. Skipping classification step.")
            grid_classi = None
            y_pred_classi = np.repeat(y_test_bin.unique(), len(y_test_bin))  
        else:
            print("Invalid classification model")

        mask_pos_train = y_combined_actual > 0 
        X_combined_1 = X_combined.loc[mask_pos_train].reset_index(drop=True)
        y_combined_1 = y_combined.loc[mask_pos_train].reset_index(drop=True)

        if mask_pos_train.sum() == 0:
            print("No positive cases in train available for regression after filtering.")
        
        n_val_filtered = mask_pos_train[:len(X_train)].sum()
        n_train_filtered = mask_pos_train.iloc[len(X_train):].sum() 
        split_indices_1 = np.zeros(len(X_combined_1))
        split_indices_1[:n_train_filtered] = -1   
        split_indices_1[n_train_filtered:] = 0    
        pds_1 = PredefinedSplit(test_fold=split_indices_1)

        if model_type == "hrf":
            reg_model = RandomForestRegressor(oob_score=True, random_state=42)

            param_grid = {
                        'max_depth': [3, 5, 7, 9],
                        'min_samples_split': [10, 50, 75, 100, 150],
                        'ccp_alpha': [0, 1 /  10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10], 
                        "criterion": ["squared_error", "absolute_error"]
            }

            if n_train_filtered == 0 or n_val_filtered == 0:
                    print(f"WARNING: No positive cases in some set (n_train={n_train_filtered}, n_val={n_val_filtered}). Using cv=5.")
                    cv_for_reg = 5
            else:
                    # Create PredefinedSplit with both train and test portions
                    split_indices_1 = np.zeros(len(X_combined_1))
                    split_indices_1[:n_train_filtered] = -1
                    split_indices_1[n_train_filtered:] = 0
                    pds_1 = PredefinedSplit(test_fold=split_indices_1)
                    pds_1 = pds_1
                    print(f"Using PredefinedSplit: n_train={n_train_filtered}, n_val={n_val_filtered}")

            grid_reg = GridSearchCV(
                estimator=reg_model, 
                param_grid=param_grid, 
                cv=pds_1, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1
            )
            
            grid_reg.fit(X_combined_1, y_combined_1)
            reg_model = grid_reg.best_estimator_
            y_pred_reg = grid_reg.predict(X_test)

        elif model_type == "hxgb":
            reg_model = XGBRegressor(random_state=42)
            param_grid = {
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.1, 0.3, 0.05, 0.01],
                'n_estimators': [10, 50, 75, 100, 150], 
                "eval_metric": ["rmse", "mae"],
                "reg_lambda": [0, 1 / 10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10]
            }

            if n_train_filtered == 0 or n_val_filtered == 0:
                print(f"WARNING: No positive cases in some set (n_train={n_train_filtered}, n_val={n_val_filtered}). Using cv=5.")
                pds_1 = 5
            else:
                # Create PredefinedSplit with both train and test portions
                split_indices_1 = np.zeros(len(X_combined_1))
                split_indices_1[:n_train_filtered] = -1
                split_indices_1[n_train_filtered:] = 0
                pds_1 = PredefinedSplit(test_fold=split_indices_1)
                print(f"Using PredefinedSplit: n_train={n_train_filtered}, n_val={n_val_filtered}")
            
            grid_reg = GridSearchCV(
                reg_model,
                param_grid,
                cv=pds_1,
                n_jobs=-1,
                verbose=0,
                scoring="neg_mean_squared_error",
            )
            grid_reg.fit(X_combined_1, y_combined_1)
            reg_model = grid_reg.best_estimator_
            y_pred_reg = grid_reg.predict(X_test)

        else:
            print("Invalid regression model")

        y_test_actual = np.exp(y_test) - epsilon
        y_pred_reg = np.exp(y_pred_reg) - epsilon
        y_pred_reg = np.where(y_pred_classi == 0, 0, y_pred_reg)

        results = pd.DataFrame({
            "actual": y_test_actual,
            "pred": y_pred_reg,
            "week_canton": test["week_canton"].values,
        })

        results.to_csv(f"../../data/model_results/results_{model_type}_{self.canton}.csv", index=False)

        if grid_classi != None:
            joblib.dump(grid_classi, f'../../models/saved_models/{model_type}_classi_{self.canton}.joblib')

        joblib.dump(grid_reg, f'../../models/saved_models/{model_type}_reg_{self.canton}.joblib')

        print(f"{model_type} ready for {self.canton}")

    def ticks_years_top(self, ax, n):
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        for i, tick in enumerate(ax.xaxis.get_major_ticks()):
            if i % n != 0:
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.gridline.set_visible(False) 

    def calculate_nrmse(self, model_type):
        results = pd.read_csv(f'../../data/model_results/results_{model_type}_{self.canton}.csv')

        rmse = root_mean_squared_error(results["actual"], results["pred"])

        nrmse = rmse / np.mean(results["actual"])

        return self.canton, nrmse

    def confint_nrmse(self, model_type, X_combined, y_combined, X_test, y_test, epsilon, n_bootstraps = 100):

        try:
            grid_classi = joblib.load(f'../../models/saved_models/{model_type}_classi_{self.canton}.joblib')
        except FileNotFoundError: 
            grid_classi = None

        try:
            grid_reg = joblib.load(f'../../models/saved_models/{model_type}_reg_{self.canton}.joblib')
        except FileNotFoundError: 
            grid_reg = joblib.load(f'../../models/saved_models/{model_type}_{self.canton}.joblib')
        
        best_params_reg = grid_reg.best_params_

        y_test = np.exp(y_test) - epsilon

        X_arr = X_combined
        y_arr = y_combined

        bs = CircularBlockBootstrap(
            52,
            X_arr,
            y_arr,
            seed = 42
        )

        models = []
        models_classi = []

        if grid_classi == None and model_type == "hrf":
                print("Only one class present in test set for classification. Skipping classification step.")
        elif grid_classi == None and model_type == "hxgb":
                print("Only one class present in test set for classification. Skipping classification step.")

        for data in bs.bootstrap(n_bootstraps):

            X_boot = data[0][0]
            y_boot = data[0][1]

            if model_type == "rf":
                modelo = RandomForestRegressor(**best_params_reg, random_state = 42)
                modelo.fit(X_boot, y_boot)
                models.append(modelo)
            elif model_type == "xgb":
                modelo = XGBRegressor(**best_params_reg, random_state = 42) 
                modelo.fit(X_boot, y_boot)
                models.append(modelo)
            elif model_type == "hrf":
                y_boot_actual = np.exp(y_boot) - epsilon
                y_boot_bin = pd.Series((y_boot_actual > 0).astype(int))

                if grid_classi != None:
                    best_params_classi = grid_classi.best_params_
                    modelo_classi = RandomForestClassifier(**best_params_classi, random_state = 42)
                    modelo_classi.fit(X_boot, y_boot_bin)
                    models_classi.append(modelo_classi)

                mask_pos_train = y_boot_actual > 0 
                X_boot_1 = X_boot[mask_pos_train]
                y_boot_1 = y_boot[mask_pos_train]
                modelo_reg = RandomForestRegressor(**best_params_reg, random_state = 42)
                modelo_reg.fit(X_boot_1, y_boot_1)
                models.append(modelo_reg)
            elif model_type == "hxgb":
                y_boot_actual = np.exp(y_boot) - epsilon
                y_boot_bin = pd.Series((y_boot_actual > 0).astype(int))

                if grid_classi != None:
                    best_params_classi = grid_classi.best_params_
                    modelo_classi = XGBClassifier(**best_params_classi, random_state = 42)
                    modelo_classi.fit(X_boot, y_boot_bin)
                    models_classi.append(modelo_classi)

                mask_pos_train = y_boot_actual > 0 
                X_boot_1 = X_boot[mask_pos_train]
                y_boot_1 = y_boot[mask_pos_train]
                modelo_reg = XGBRegressor(**best_params_reg, random_state = 42)
                modelo_reg.fit(X_boot_1, y_boot_1)
                models.append(modelo_reg)

        if grid_classi == None:                     
            predictions = np.array([model.predict(X_test) for model in models])
            predictions = np.exp(predictions) - epsilon
        else: 
            predictions_classi = np.array([model.predict(X_test) for model in models_classi])
            predictions_reg = np.array([model.predict(X_test) for model in models])
            predictions_reg = np.exp(predictions_reg) - epsilon
            predictions = np.array([np.where(y_pred_classi == 0, 0, y_pred_reg) for y_pred_classi, y_pred_reg in zip(predictions_classi, predictions_reg)])


        nrmse = np.array([root_mean_squared_error(y_true = y_test, y_pred = pred) / np.mean(y_test) for pred in predictions])

        y_pred_point = np.mean(nrmse)

        lower_bound = np.percentile(nrmse, 2.5)
        upper_bound = np.percentile(nrmse, 97.5)

        print(f"point estimate: {round(y_pred_point, 2)}")
        print(f"lower bound: {round(lower_bound, 2)}, upper bound: {round(upper_bound, 2)}")

    