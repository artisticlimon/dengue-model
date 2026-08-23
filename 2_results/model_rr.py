"""
This script contains the Model class used to train and evaluate the different models for relative risk. It includes methods for partitioning the data, training the models, calculating variable importance, calculating prediction intervals, and calculating confidence intervals for the NRMSE metric. The class is designed to be flexible and can be used for different cantons and model types (Random Forest, XGBoost, Hybrid Random Forest, Hybrid XGBoost).
"""

import pandas as pd
from sklearn.metrics import root_mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit, TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor
import joblib
from sklearn.inspection import permutation_importance
from arch.bootstrap import IIDBootstrap
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.subsample import BlockBootstrap
import time

class Model:

    """
    Attributes:
    ----
    df: dataframe
        complete dataset with, including lags and target variable, with each row being a week-canton pair
    canton: str
        the canton for which we want to train and evaluate the model. 
    """

    def __init__(self, df, canton):
    
        """
        Method that initializes the class, filters for the desired canton, and drops the columns that are not needed for the models because they correspond to that same week. 

        Parameters
        ----
        df: dataframe
            complete dataset with, including lags and target variable, with each row being a week-canton pair
        canton: str
            the canton for which we want to train and evaluate the model.
        Returns
        ----
            Nothing, but initializes the class with the filtered and cleaned dataframe for the desired canton.
        """
        self.df = df.copy()
        self.canton = canton
        if canton != "full":
            self.df = df[df["week_canton"].str.contains(canton)] # Filter for the desired canton
        else:
            self.df = df    
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

        """
        Method that drops the lag columns that are not needed for the models, depending on which time frame we want to use to make predictions (usual options include immediate, medium term, all lags).

        Parameters
        ----
        i: int
            initial month of lags to drop (e.g., if we want to keep lags 1-4, i would be 5)
        f: str
            last month of lags to drop (e.g., if we want to keep lags 1-4, f would be 9)
        df_temp: dataframe
            dataframe from which to drop the lag columns
        Returns
        ----
            df_temp: dataframe with the specified lag columns dropped
        """

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

    def partition(self, epsilon = 0.2):

        """
        Method that created that partitions the dataset into train, validation, and test sets for relative risk as a target value. It drops other target values and relative risk lags. It adds a small value (epsilon) to the relative risk before log-transforming it, to avoid issues with zero values. Finally, it also creates a predefined split object to be used for both hyperparameter tuning and RFECV, specifying which data points belong to train and which to validation.

        Returns
        ----
            X_combined: dataframe
                dataframe with the features for the train and validation set
            y_combined: pandas Series
                pandas Series with the target variable (relative risk) for the train and validation set, log-transformed and with epsilon added
            X_train: dataframe
                dataframe with the features for the train set. It is returned separately from X_combined to be used to create the new predefined splits for the regression part of the hybrid model.
            X_test: dataframe
                dataframe with the features for the test set
            y_test: pandas Series
                pandas Series with the target variable (relative risk) for the test set, log-transformed and with epsilon added
            test: dataframe
                dataframe with the test set, including the week_canton column for plotting purposes
            pds: predefined split
                the predefined split object that specifies which data points belong to train and which to validation, to be used for both hyperparameter tuning and RFECV.
            epsilon: int
                the small value that was added to the relative risk before log-transforming it, to avoid issues with zero values. The default value was found through trial and error, looking at histograms for each value.
        """

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

        # 2020-2023 as train, 2024 as validation, 2025 as test  
        train = df_temp[df_temp['week_canton'] < '2024-1-101']
        val = df_temp[(df_temp['week_canton'] >= '2024-1-101') & (df_temp['week_canton'] < '2025-1-101')]
        test = df_temp[df_temp['week_canton'] >= '2025-1-101'] 

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

        pds = PredefinedSplit(test_fold=split_indices) # Create predefined split to be used for both hyperparameter tuning and RFECV, specifying which data points belong to train and which to validation

        return X_combined, y_combined, X_train, X_test, y_test, test, pds, epsilon

    def serie_temp_canton(self, model_type, X_combined, y_combined, y_test, lower_bound, upper_bound, epsilon, lims = [0, 7]):

        """
        Method that plots the time series of actual and predicted relative risk values for the test set, including the confidence intervals for the predictions. It reads the results from the csv file where they were saved after making predictions with the trained model.

        Parameters
        ----
        model_type: str
            the type of model for which to plot the results (options: "rf", "xgb", "hrf", "hxgb")
        X_combined, y_combined: dataframe, Pandas Series
            training and validation data
        y_test: pandas series
            test relative risk data
        lower_bound: array-like
            array-like with the lower bound of the confidence intervals for the predictions, to be used as the lower limit shaded area around the predicted values in the plot.
        upper_bound: array-like
            array-like with the upper bound of the confidence intervals for the predictions, to be used as the upper limit shaded area around the predicted values in the plot.
        epsilon: float
            value used before log-transform
        lims: list
            list with the limits for the y-axis of the plot, to be used as the ylim parameter in the plot. 
        Returns
        ----
            Nothing, just shows the plot with the time series of actual and predicted relative risk values for the test set, including the confidence intervals for the predictions.
        """
        preds = pd.read_csv(f'../data/model_results/results_{model_type}_full.csv')
        results = preds[preds["week_canton"].str.contains(self.canton)] 

        df_copy = self.df.copy()

        df_copy['week_canton'] = df_copy['week_canton'].str.extract(r'(\d{4}-\d+)').iloc[:, 0].tolist()

        y_pred_train = self.calculate_train_predictions(model_type, X_combined, epsilon)

        y_combined = np.exp(y_combined) - epsilon
        y_test = np.exp(y_test) - epsilon
        
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(df_copy['week_canton'].iloc[0:len(y_combined)], y_combined, label='Real', marker='o', linestyle = "dashed", color = "#000000", linewidth=4)
        ax.plot(df_copy['week_canton'].iloc[len(y_combined):], y_test, marker='o', linestyle = "dashed", color = "#000000", linewidth=4)
        ax.plot(df_copy['week_canton'].iloc[len(y_combined):], results["pred"], label='Predicted', marker='o',linestyle = "solid", color = "#009E73", linewidth=4)
        ax.plot(df_copy['week_canton'].iloc[0:len(y_combined)], y_pred_train, marker='o', linestyle = "solid", color = "#009E73", linewidth=4)
        ax.fill_between(x = df_copy['week_canton'].iloc[len(y_combined):], label = "CI (95%)", y1 = lower_bound, y2 = upper_bound, alpha = 0.2, color = "#009E73")
        ax.legend(loc = "upper right", fontsize = 18)
        y_ticks = list(range(0, int(np.ceil(lims[1]))))
        ax.set_yticks(y_ticks)
        y_labels = [str(tick) if i % 2 == 0 else "" for i, tick in enumerate(y_ticks)]
        ax.set_yticklabels(y_labels, fontsize=14, fontweight="bold")
        self.ticks_years_top(ax, 25)
        ax.set_xlabel('Week', fontweight = "bold", fontsize = 20)
        ax.set_ylabel(f'Relative risk', fontweight = "bold", fontsize = 20)
        ax.set_ylim(lims)
        ax.set_xlim([0, 304])

        # ax.set_title(f"{model_type} for {self.canton}")
        plt.show()

    def custom_threshold_score(self, y_true, y_pred_proba, threshold):

        """
        Scorer used to calculate the permutation importance for the classification part of the hybrid models. It uses a custom threshold to convert predicted probabilities into binary predictions, and then calculates the negative root mean squared error between the true labels and the predicted labels.
        
        Parameters
        ----
        y_true: Pandas series
            True labels for the classification task.
        y_pred_proba: array-like
            Predicted probabilities for the positive class.
        threshold: float
            Threshold to convert predicted probabilities into binary predictions.
        """
    
        if y_pred_proba.ndim == 2:
            y_scores = y_pred_proba[:, 1]
        else:
            # If it is already 1-dimensional, use it directly
            y_scores = y_pred_proba
        
        y_pred= (y_scores >= threshold).astype(int)
        
        return -1 * (root_mean_squared_error(y_true, y_pred)) ** 2 # The neg root mean squared error is used as the scoring metric for the permutation importance, to be consistent with the scoring used in the regression part of the hybrid models.
 
    
    def var_importance(self, X_test, y_test, model_type, repeats = 100):

        """
        Method that calculates and plots the permutation importance of the features for the specified model type, using the test set. It reads the trained model from the saved models folder, calculates the permutation importance using the sklearn function, and then creates a horizontal bar plot with the importance scores for the top 10 features. If it is a hurdle model, it calculates the permutation importance for both the classification and regression parts of the model, and plots them separately.

        Parameters
        ----
        X_test: dataframe
            dataframe with the features for the test set
        y_test: pandas Series
            pandas Series with the target variable (relative risk) for the test set, log-transformed and with epsilon added
        model_type: str
            the type of model for which to calculate the variable importance (options: "rf", "xgb", "hrf", "hxgb")
        repeats: int
            the number of times to permute a feature for calculating the permutation importance, to be used as the n_repeats parameter in the sklearn function. 

        Returns
        ----
            imp_df: dataframe
                dataframe with the features, their importance scores, and their standard deviation, sorted by importance score in descending order.
            It also shows the horizontal bar plot with the importance scores for the top 10 features.
        """
        start = time.perf_counter()

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
        try:
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_reg_{self.canton}.joblib')
        except FileNotFoundError: 
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_{self.canton}.joblib')
        
        regressor = grid_reg.best_estimator_

        if model_type == "hrf" or model_type == "hxgb":
            try:
                grid_classi = joblib.load(f'../models/saved_models/{model_type}_classi_{self.canton}.joblib')
            except FileNotFoundError:
                grid_classi = None
        
            if self.canton == "full":
                thresholds = pd.read_csv(f"../data/model_results/thresholds/full_threshold_{model_type}.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
                best_threshold = thresholds[thresholds["canton"] == self.canton]["best_th"].values[0]
                classifier = None if grid_classi is None else grid_classi.best_estimator_
                regressor = grid_reg.best_estimator_
            else: 
                thresholds = pd.read_csv(f"../data/model_results/thresholds/thresholds_{model_type}.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
                best_threshold = thresholds[thresholds["canton"] == self.canton]["best_th"].values[0]
                classifier = None if grid_classi is None else grid_classi.best_estimator_
                regressor = grid_reg.best_estimator_ 

            print("Classifier")

            if grid_classi != None:
                scorer = make_scorer(
                        self.custom_threshold_score,
                        response_method="predict_proba",
                        threshold=best_threshold
                )

                r_classi = permutation_importance(classifier, X_test, y_test, n_repeats=repeats, random_state=42, scoring=scorer, n_jobs = -1)
                
                imp_df_classi = pd.DataFrame({"Feature": X_test.columns, f"imp_{model_type}": r_classi.importances_mean, f"std_{model_type}": r_classi.importances_std}).sort_values(by=f"imp_{model_type}", ascending=False).reset_index(drop=True)

                fig, ax = plt.subplots(figsize=(12, 8))
                imp_df_classi_sorted = imp_df_classi.head(10).sort_values(by=f"imp_{model_type}", ascending=True)
                ax.barh(imp_df_classi_sorted["Feature"], imp_df_classi_sorted[f"imp_{model_type}"])
                ax.tick_params(axis='y', labelsize=14)
                for label in ax.get_yticklabels():
                    label.set_fontweight('bold')
                ax.tick_params(axis='x', labelsize=14)
                for label in ax.get_xticklabels():
                    label.set_fontweight('bold')
                ax.bar_label(ax.containers[0], fmt='%.2e', fontsize = 10, fontweight = "bold")
                ax.set_xlabel("Importance", fontsize = 20, fontweight = "bold")
                # fig.suptitle(f"{model_type} classification permutation importance for RR in {self.canton}")
                plt.show()

            print("Regressor")

        r = permutation_importance(regressor, X_test, y_test, n_repeats=repeats, random_state=42, scoring='neg_mean_squared_error', n_jobs = -1)

        imp_df = pd.DataFrame({"Feature": X_test.columns, f"imp_{model_type}": r.importances_mean, f"std_{model_type}": r.importances_std}).sort_values(by=f"imp_{model_type}", ascending=False).reset_index(drop=True)

        print(imp_df)

        fig, ax = plt.subplots(figsize=(12, 8))
        imp_df_sorted = imp_df.head(10).sort_values(by=f"imp_{model_type}", ascending=True)
        ax.barh(imp_df_sorted["Feature"], imp_df_sorted[f"imp_{model_type}"])
        ax.tick_params(axis='y', labelsize=14)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.tick_params(axis='x', labelsize=14)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        ax.bar_label(ax.containers[0], fmt='%.2e', fontsize = 10, fontweight = "bold")
        ax.set_xlabel("Importance", fontsize = 20, fontweight = "bold")
        # fig.suptitle(f"{model_type} regression permutation importance for {var} in {self.canton}")
        plt.show()

        end = time.perf_counter()

        print(f"Elapsed time: {end - start:.2f} seconds")

        return imp_df

    def prediction_intervals_aci(self,
        model_type,
        X_train,
        y_train,
        X_test,
        y_test,
        epsilon,
        alpha=0.05,
        gamma=0.005,
        n_resamplings=100
    ):
        """
        This method uses Adaptive Conformal Inference (ACI) to construct prediction intervals.

        Parameters
        ----------
        model_type: str
            the type of model, which can be "rf", "xgb", "hrf", or "hxgb"
        X_train, y_train: dataframe, Pandas series
            training data
        X_test, y_test: dataframe, Pandas series
            test data (y_test is only used for online updating)
        epsilon: float
            value used before log-transform
        alpha: float
            miscoverage level (0.05 -> 95% PI)
        gamma: float 
            adaptation rate
        n_resamplings: 
            number of bootstrap resamples

        Returns
        -------
        y_pred: array-like
            point predictions
        lower: array-like
            lower prediction interval
        upper: array-like
            upper prediction interval
        """

        start = time.perf_counter()

        try:
            grid = joblib.load(f'../models/saved_models/{model_type}_reg_full.joblib')
        except: 
            grid = joblib.load(f'../models/saved_models/{model_type}_full.joblib')
        
        best_reg = grid.best_estimator_

        X_train = X_train.to_numpy(dtype=np.float32)
        X_test = X_test.to_numpy(dtype=np.float32)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        cv = BlockBootstrap(
            n_resamplings=n_resamplings,
            length=1,
            overlapping=False,
            random_state=42,
        )

        mapie = TimeSeriesRegressor(
            estimator=best_reg,
            method="aci",
            cv=cv,
            conformity_score=AbsoluteConformityScore(sym=True),
            agg_function="mean",
            random_state=42,
            n_jobs=-1,
        )

        mapie.fit(X_train, y_train)

        lower = []
        upper = []

        for x, y in zip(X_test, y_test):

            pred, interval = mapie.predict(
                x.reshape(1, -1),
                confidence_level=1 - alpha,
                ensemble=True,
                optimize_beta=False,
            )

            lower.append(interval[0, 0, 0])
            upper.append(interval[0, 1, 0])

            mapie.adapt_conformal_inference(
                x.reshape(1, -1),
                np.array([y]),
                gamma=gamma,
                ensemble=True,
                optimize_beta=False,
            )

        lower = np.exp(lower) - epsilon
        upper = np.exp(upper) - epsilon

        end = time.perf_counter()
        
        print(f"Elapsed time: {end - start:.2f} seconds")

        return (
            np.asarray(lower),
            np.asarray(upper),
        )
    
    def rf_reg(self, X_combined, y_combined, X_test, y_test, test, pds, epsilon):

        """
        Method that trains a random forest model for the class instance, performing a grid search to tune hyperparameters. It also saves the resulting predictions and the trained model in a separate folder.

        Parameters
        ----
        X_combined: dataframe
            dataframe with the features for the combined train and validation set

        y_combined: pandas Series
            pandas Series with the target variable (relative risk) for the combined train and validation set, log-transformed and with epsilon added
        
        X_test: dataframe
            dataframe with the features for the test set 
        
        y_test: pandas Series
            pandas Series with the target variable (relative risk) for the test set, log-transformed and with epsilon added

        test: dataframe
            dataframe with the test set, including the week_canton column for so that it can be used to save the results linked to a week.
        
        pds: predefined split
            the predefined split object that specifies which data points belong to train and which to validation, to be used for both hyperparameter tuning.
        
        epsilon: int
            the small value that was added to the relative risk before log-transforming it, to avoid issues with zero values. The default value was found through trial and error, looking at histograms for each value. 

        Returns
        ----
            Nothing, but it saves the model results and the trained model, and prints when it is ready.
        """

        start = time.perf_counter()

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

        results_rf.to_csv(f'../data/model_results/results_rf_{self.canton}.csv')

        joblib.dump(grid_rf, f'../models/saved_models/rf_{self.canton}.joblib')

        print(f"RF Model Ready for {self.canton}")

        end = time.perf_counter()
        
        print(f"Elapsed time: {end - start:.2f} seconds")

        elapsed_time = end - start

        return elapsed_time

    def xgb_reg(self, X_combined, y_combined, X_test, y_test, test, pds, epsilon):

        """
        Method that trains a XGBoost model for the class instance, performing a grid search to tune hyperparameters. It also saves the resulting predictions and the trained model in a separate folder.

        Parameters
        ----
        X_combined: dataframe
            dataframe with the features for the combined train and validation set

        y_combined: pandas Series
            pandas Series with the target variable (relative risk) for the combined train and validation set, log-transformed and with epsilon added
        
        X_test: dataframe
            dataframe with the features for the test set 
        
        y_test: pandas Series
            pandas Series with the target variable (relative risk) for the test set, log-transformed and with epsilon added

        test: dataframe
            dataframe with the test set, including the week_canton column for so that it can be used to save the results linked to a week.
        
        pds: predefined split
            the predefined split object that specifies which data points belong to train and which to validation, to be used for both hyperparameter tuning.
        
        epsilon: int
            the small value that was added to the relative risk before log-transforming it, to avoid issues with zero values. The default value was found through trial and error, looking at histograms for each value. 

        Returns
        ----
            Nothing, but it saves the model results and the trained model, and prints when it is ready.
        """
        start = time.perf_counter()

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

        results_xgb.to_csv(f'../data/model_results/results_xgb_{self.canton}.csv')

        joblib.dump(grid_xgb, f'../models/saved_models/xgb_{self.canton}.joblib')

        print(f"XGB Model Ready for {self.canton}")

        end = time.perf_counter()
        
        print(f"Elapsed time: {end - start:.2f} seconds")

        elapsed_time = end - start
        
        return elapsed_time

    def hybrid(self, model_type, X_combined, y_combined, X_train, X_test, y_test, test, pds, epsilon):

        """
        Method that trains a hybrid model (Random Forest-Random Forest or XGBoost-XGBoost) for the class instance, performing a grid search to tune hyperparameters. It also saves the resulting predictions and the trained model in a separate folder.

        Parameters
        ----
        model_type: str
            this indicates which of the hybrid models are desired ("hrf" for Random Forest Hybrid and "hxgb" for XGBoost hybrid)

        X_combined: dataframe
            dataframe with the features for the combined train and validation set

        y_combined: pandas Series
            pandas Series with the target variable (relative risk) for the combined train and validation set, log-transformed and with epsilon added
        
        X_train: dataframe
            dataframe with the features for the train set, to be used when defining the second validation predefined split for the regression model
            
        X_test: dataframe
            dataframe with the features for the test set 
        
        y_test: pandas Series
            pandas Series with the target variable (relative risk) for the test set, log-transformed and with epsilon added

        test: dataframe
            dataframe with the test set, including the week_canton column for so that it can be used to save the results linked to a week.
        
        pds: predefined split
            the predefined split object that specifies which data points belong to train and which to validation, to be used for both hyperparameter tuning.
        
        epsilon: int
            the small value that was added to the relative risk before log-transforming it, to avoid issues with zero values. The default value was found through trial and error, looking at histograms for each value. 

        Returns
        ----
            best_threshold: float
                return the best threshold so that it can be saved to calculate intervals
        """
        start = time.perf_counter()

        X_combined = X_combined.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_combined = pd.Series(y_combined).reset_index(drop=True)
        y_test = pd.Series(y_test).reset_index(drop=True)
        test = test.reset_index(drop=True)

        y_combined_actual = np.exp(y_combined) - epsilon
        y_test_actual = np.exp(y_test) - epsilon 
        y_train_binary = pd.Series((y_combined_actual > 0).astype(int)).reset_index(drop=True) 
        y_test_binary = pd.Series((y_test_actual > 0).astype(int)).reset_index(drop=True) 
        if y_train_binary.nunique() < 2:
            print(f"WARNING: train set only contains one label: {y_train_binary.unique()[0]}")

        if model_type == "hrf" and y_train_binary.nunique() >= 2:
            clf = RandomForestClassifier(random_state=42)
            param_grid = {
                "max_depth": [3, 5, 7, 9],
                "min_samples_split": [10, 50, 75, 100, 150],
                "ccp_alpha": [0, 1 / 10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10],
                "criterion": ["gini", "entropy"],
                "class_weight": [None, "balanced"]
            }
            grid_classi = GridSearchCV(clf, param_grid, cv=pds, n_jobs=-1, verbose=0, scoring="precision")
            grid_classi.fit(X_combined, y_train_binary)

        elif model_type == "hxgb" and y_train_binary.nunique() >= 2:
            clf = XGBClassifier(random_state=42)
            param_grid = {
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.1, 0.3, 0.05, 0.01],
                "n_estimators": [10, 50, 75, 100, 150],
                "reg_lambda": [0, 1 / 10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10],
                "eval_metric": ["logloss"]
            }
            grid_classi = GridSearchCV(clf, param_grid, cv=pds, n_jobs=-1, scoring="precision")
            grid_classi.fit(X_combined, y_train_binary)  
        
        elif y_test_binary.nunique() < 2:
            print("Only one class present in train set for classification. Skipping classification step.")
            grid_classi = None
            y_pred_classi = np.repeat(y_test_binary.unique(), len(y_test_binary))  
        else:
            print("Invalid classification model")

        mask_pos_train = y_combined_actual > 0 
        X_combined_1 = X_combined.loc[mask_pos_train].reset_index(drop=True)
        y_combined_1 = y_combined.loc[mask_pos_train].reset_index(drop=True)

        # If all cases are 0, skip
        if mask_pos_train.sum() == 0:
            print("No positive cases in train available for regression after filtering.")
            y_test_actual = np.exp(y_test) - epsilon
            y_pred_reg = np.zeros(len(X_test))

            results = pd.DataFrame({
                "actual": y_test_actual,
                "pred": y_pred_reg,
                "week_canton": test["week_canton"].values
            })

            results.to_csv(
                 f"../data/model_results/results_{model_type}_{self.canton}.csv",
                 index=False
            )

            print(f"{model_type} ready for {self.canton}")

            return None
        
        # Create predefined split to differentiate between train and validation points when the model is being tuned
        n_train_filtered = mask_pos_train[:len(X_train)].sum()
        n_val_filtered = mask_pos_train[len(X_train):].sum()

        split_indices_1 = np.empty(len(X_combined_1))
        split_indices_1[:n_train_filtered] = -1
        split_indices_1[n_train_filtered:] = 0

        pds_1 = PredefinedSplit(test_fold=split_indices_1)

        if model_type == "hrf":
            reg_model = RandomForestRegressor(random_state=42)

            param_grid = {
                        'max_depth': [3, 5, 7, 9],
                        'min_samples_split': [10, 50, 75, 100, 150],
                        'ccp_alpha': [0, 1 /  10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10], 
                        "criterion": ["squared_error", "absolute_error"]
            }

            if n_val_filtered == 0:
                print(
                    "No positive validation observations after filtering. "
                    "Skipping hyperparameter tuning."
                )
                pds_1 = TimeSeriesSplit(n_splits=5)
            else:
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
            y_pred_reg = reg_model.predict(X_test)

        elif model_type == "hxgb":
            reg_model = XGBRegressor(random_state=42)
            param_grid = {
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.1, 0.3, 0.05, 0.01],
                'n_estimators': [10, 50, 75, 100, 150], 
                "eval_metric": ["rmse", "mae"],
                "reg_lambda": [0, 1 / 10**5, 1 / 10**4, 1 / 10**3, 0.01, 0.1, 1, 10]
            }

            if n_val_filtered == 0:
                print("No positive validation observations after filtering. Skipping hyperparameter tuning.")
                pds_1 = TimeSeriesSplit(n_splits=5)
                
            else:
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
            y_pred_reg = reg_model.predict(X_test)

        else:
            print("Invalid regression model")

        if grid_classi != None:
            best_classi = grid_classi.best_estimator_

            split_indices = pds.test_fold
            val_mask = split_indices == 0
            X_val = X_combined.loc[val_mask].reset_index(drop=True)
            y_val = y_combined.loc[val_mask].reset_index(drop=True)

            thresholds = [x / 100 for x in range(0, 101)]
            probabilities_val = best_classi.predict_proba(X_val)[:, 1]
            y_test_actual = np.exp(y_test) - epsilon
            val_pred_reg = reg_model.predict(X_val)
            val_pred_reg_actual = np.exp(val_pred_reg) - epsilon
            y_val_actual = np.exp(y_val) - epsilon

            best_threshold = None
            best_nrmse = np.inf

            for threshold in thresholds:
                y_pred_classi = (probabilities_val >= threshold).astype(int)
                y_pred_reg_temp = np.where(y_pred_classi == 0, 0, val_pred_reg_actual)
                rmse_temp = root_mean_squared_error(y_val_actual, y_pred_reg_temp)
                nrmse_temp = rmse_temp / (np.mean(y_val_actual) + epsilon)
                if nrmse_temp < best_nrmse:
                    best_nrmse = nrmse_temp
                    best_threshold = threshold

        if grid_classi != None:
            probabilities_test = best_classi.predict_proba(X_test)[:, 1]
            y_pred_classi = (probabilities_test >= best_threshold).astype(int)
            y_pred_reg = np.exp(y_pred_reg) - epsilon
            y_pred_reg = np.where(y_pred_classi == 0, 0, y_pred_reg)
        else:
            best_threshold = None
            y_test_actual = np.exp(y_test) - epsilon
            y_pred_reg = np.exp(y_pred_reg) - epsilon

        results = pd.DataFrame({
            "actual": y_test_actual,
            "pred": y_pred_reg,
            "week_canton": test["week_canton"].values
        })

        results.to_csv(f"../data/model_results/results_{model_type}_{self.canton}.csv", index=False)

        # Only save classification model if the classification step was not skipped due to only one class being present in test set
        if grid_classi != None:
             joblib.dump(grid_classi, f'../models/saved_models/{model_type}_classi_{self.canton}.joblib')

        joblib.dump(grid_reg, f'../models/saved_models/{model_type}_reg_{self.canton}.joblib')

        print(f"{model_type} ready for {self.canton}")

        end = time.perf_counter()

        elapsed_time = end - start
        
        print(f"Elapsed time: {end - start:.2f} seconds")

        return best_threshold, elapsed_time

    def ticks_years_top(self, ax, n):

        """
        Method that adjusts plot ticks in the time series plot so that it looks more organized. 

        Parameters
        ----
        ax: Axes
            canvas were the data will be plotted

        n: int
            show every n-th tick

        Returns
        ----
            Nothing, but adds neat ticks to the plot
        """
        ax.tick_params(axis='x', labelsize=12)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')

        ax.xaxis.set_major_locator(ticker.MultipleLocator(n))

    def calculate_train_predictions(self, model_type, X_combined, epsilon): 

        """
        Method that calculates the predictions for the training set.

        Parameters
        ----
        model_type: str
            model for which the predictions will be calculated (options: "rf", "xgb", "hrf", "hxgb")
        X_combined: dataframe
            training and validation data
        epsilon: float
            the small value that was added to the relative risk before log-transforming.

        Returns
        ----
            predictions: array-like
                the calculated predictions for the training set
        """
        # Check if there's a classifier model if it's a hybrid model type
        try:
            grid_classi = joblib.load(f'../models/saved_models/{model_type}_classi_{self.canton}.joblib')
            modelo_classi = grid_classi.best_estimator_
        except FileNotFoundError: 
            grid_classi = None

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
        try:
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_reg_{self.canton}.joblib')
            modelo_reg = grid_reg.best_estimator_
        except FileNotFoundError: 
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_{self.canton}.joblib')
            modelo_reg = grid_reg.best_estimator_

        if model_type == "hrf":
            if self.canton == "full":
                thresholds = pd.read_csv(f"../data/model_results/thresholds/full_threshold_hrf.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            else: 
                thresholds = pd.read_csv(f"../data/model_results/thresholds/thresholds_hrf.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == self.canton]["best_th"].values[0]
        elif model_type == "hxgb":
            if self.canton == "full":
                thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hxgb.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            else: 
                thresholds = pd.read_csv("../data/model_results/thresholds/thresholds_hxgb.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == self.canton]["best_th"].values[0]

        if grid_classi == None:                     
            predictions = modelo_reg.predict(X_combined)
            predictions = np.exp(predictions) - epsilon
        else: 
            y_pred_reg = modelo_reg.predict(X_combined)
            y_pred_reg = np.exp(y_pred_reg) - epsilon
            prob_classi = modelo_classi.predict_proba(X_combined)[:, 1]
            predictions = np.where(prob_classi < best_threshold, 0, y_pred_reg)

        return predictions

    def calculate_train_nrmse(self, model_type, X_combined, y_combined, epsilon):
        """
        Method that calculates the NRMSE for the predictions from a model of a certain type, using the training set.

        Parameters
        ----
        model_type: str
            model for which the NRMSE will be calculated (options: "rf", "xgb", "hrf", "hxgb")
        X_combined, y_combined: dataframe, pandas Series
            training and validation data (y_combined is log-transformed and with epsilon added)
        epsilon: float
            the small value that was added to the relative risk before log-transforming.
        
        Returns
        ----
            self.canton: str
                the canton included in the class instance

            nrmse: float
                the calculated NRMSE
        """

        predictions = self.calculate_train_predictions(self, model_type, X_combined, epsilon)

        y_true = np.exp(y_combined) - epsilon

        rmse = root_mean_squared_error(y_true, predictions)
        
        nrmse = rmse / (np.mean(y_true) + epsilon)

        return self.canton, nrmse

    def calculate_test_nrmse(self, model_type, epsilon):

        """
        Method that calculates the NRMSE for the test set predictions from a model of a certain type.

        Parameters
        ----
        model_type: str
            model for which the NRMSE will be calculated (options: "rf", "xgb", "hrf", "hxgb")

        epsilon: float
            the small value that was added to the relative risk before log-transforming.

        Returns
        ----
            self.canton: str
                the canton included in the class instance

            nrmse: float
                the calculated NRMSE
        """
        
        results = pd.read_csv(f'../data/model_results/results_{model_type}_{self.canton}.csv')

        rmse = root_mean_squared_error(results["actual"], results["pred"])
        
        nrmse = rmse / (np.mean(results["actual"]) + epsilon)

        return self.canton, nrmse
  
    def confint_nrmse(self, model_type, X_combined, y_combined, X_test, y_test, epsilon, which_set, n_bootstraps = 500):

        """
        Method that calculates the point estimate NRMSE for the predictions from an individual model of a certain type, using either the training or test set, and calculates a 95% confidence interval for the NRMSE using bootstrapping.

        Parameters
        ----
        model_type: str
            model for which the NRMSE will be calculated (options: "rf", "xgb", "hrf", "hxgb")
        
        X_combined, y_combined: dataframe, pandas Series
            training and validation data

        X_test, y_test: dataframe, pandas Series
            test data
        
        epsilon: float
            the small value that was added to the relative risk before log-transforming.

        which_set: str
            indicates which set to use for the NRMSE calculation and bootstrapping ("train" or "test")
        
        n_bootstraps: int
            number of bootstrap resamples to use for the confidence interval calculation
        
        Returns
        ----
            self.canton: str
                the canton included in the class instance

            point: float    
                the point estimate for the NRMSE, calculated by averaging the NRMSEs from the bootstrap resamples

            lower_bound: float
                the lower bound of the 95% confidence interval for the NRMSE, calculated by taking the 2.5th percentile of the NRMSEs from the bootstrap resamples

            upper_bound: float
                the upper bound of the 95% confidence interval for the NRMSE, calculated by taking the 97.5th percentile of the NRMSEs from the bootstrap resamples
        """

        start = time.perf_counter()
        # Check if there's a classifier model if it's a hybrid model type
        try:
            grid_classi = joblib.load(f'../models/saved_models/{model_type}_classi_{self.canton}.joblib')
            modelo_classi = grid_classi.best_estimator_
        except FileNotFoundError: 
            grid_classi = None

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
        try:
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_reg_{self.canton}.joblib')
        except FileNotFoundError: 
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_{self.canton}.joblib')
        
        modelo_reg = grid_reg.best_estimator_

        if which_set == "test":
            bs = IIDBootstrap(
                X_test,
                y_test,
                seed = 42
            )
            n = len(X_test)
        elif which_set == "train":
            bs = IIDBootstrap(
                X_combined,
                y_combined,
                seed = 42
            )
            n = len(X_combined)
        else:
            print("Choose valid set")

        nrmse = []

        if grid_classi == None and model_type == "hrf":
            print("Only one class present in train set for classification. Skipping classification step.")
        elif grid_classi == None and model_type == "hxgb":
            print("Only one class present in train set for classification. Skipping classification step.")

        if model_type == "hrf":
            if self.canton == "full":
                thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hrf.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            else: 
                thresholds = pd.read_csv("../data/model_results/thresholds/thresholds/thresholds_hrf.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == self.canton]["best_th"].values[0]
        elif model_type == "hxgb":
            if self.canton == "full":
                thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hxgb.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            else: 
                thresholds = pd.read_csv("../data/model_results/thresholds/thresholds_hxgb.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == self.canton]["best_th"].values[0]

        for data in bs.bootstrap(n_bootstraps):

            X_boot = data[0][0]
            y_boot = data[0][1]

            if model_type == "rf" or model_type == "xgb":
                preds = modelo_reg.predict(X_boot)
                preds = np.exp(preds) - epsilon
            elif model_type == "hrf" or model_type == "hxgb":
                pred_reg = modelo_reg.predict(X_boot)
                pred_reg = np.exp(pred_reg) - epsilon
                prob_classi = modelo_classi.predict_proba(X_boot)[:, 1]
                preds = np.where(prob_classi < best_threshold, 0, pred_reg)

            y_true = np.exp(y_boot) - epsilon
                
            nrmse.append(root_mean_squared_error(y_true = y_true, y_pred = preds) / (np.mean(np.exp(y_boot) - epsilon) + epsilon))

        point = np.mean(nrmse)
        
        lower_bound = np.percentile(nrmse, 2.5)
        upper_bound = np.percentile(nrmse, 97.5)

        end = time.perf_counter()
        
        print(f"Elapsed time: {end - start:.2f} seconds")

        return self.canton, point, lower_bound, upper_bound
    
    def calculate_nrmse_full(self, model_type, X_combined, y_combined, X_test, y_test, epsilon, which_set):

        """
        Method that calculates the NRMSE for the predictions from the global model of a certain type, using either the training or test set

        Parameters
        ----
        model_type: str
            model for which the NRMSE will be calculated (options: "rf", "xgb", "hrf", "hxgb")
        
        X_combined, y_combined: dataframe, pandas Series
            training and validation data

        X_test, y_test: dataframe, pandas Series
            test data
        
        epsilon: float
            the small value that was added to the relative risk before log-transforming.

        which_set: str
            indicates which set to use for the NRMSE calculation and bootstrapping ("train" or "test")

        Returns
        ----
            self.canton: str
                the canton included in the class instance

            nrmse: float    
                the point estimate for the NRMSE, using the predictions from our data
        """

        # Check if there's a classifier model if it's a hybrid model type
        try:
            grid_classi = joblib.load(f'../models/saved_models/{model_type}_classi_full.joblib')
            modelo_classi = grid_classi.best_estimator_
        except FileNotFoundError: 
            grid_classi = None

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
        try:
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_reg_full.joblib')
            modelo_reg = grid_reg.best_estimator_
        except FileNotFoundError: 
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_full.joblib')
            modelo_reg = grid_reg.best_estimator_

        if model_type == "hrf":
            thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hrf.csv")
            thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == "full"]["best_th"].values[0]
        elif model_type == "hxgb":
            thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hxgb.csv")
            thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == "full"]["best_th"].values[0]

        expected_features = list(modelo_reg.feature_names_in_)
    
        urb_features = [col for col in expected_features if col.startswith("urb_")]
            
        for X in [X_combined, X_test]:
            for col in urb_features:
                if col not in X.columns:
                    X[col] = 0

        X_combined = X_combined[expected_features]
        X_test = X_test[expected_features]
        
        if which_set == "test":
            X_eval = X_test
            y_eval = y_test
        elif which_set == "train":
            X_eval = X_combined
            y_eval = y_combined
        else:
            raise ValueError("which_set must be 'test' or 'train'")
        
        if grid_classi is None:
            predictions = modelo_reg.predict(X_eval)
            predictions = np.exp(predictions) - epsilon
        else:
            y_pred_reg = modelo_reg.predict(X_eval)
            y_pred_reg = np.exp(y_pred_reg) - epsilon
            prob_classi = modelo_classi.predict_proba(X_eval)[:, 1]
            predictions = np.where(prob_classi < best_threshold, 0, y_pred_reg)

        y_true = np.exp(y_eval) - epsilon
        
        rmse = root_mean_squared_error(y_true, predictions)
        nrmse = rmse / (np.mean(y_true) + epsilon)

        return self.canton, nrmse
        
    def confint_nrmse_full(self, model_type, X_combined, y_combined, X_test, y_test, epsilon, which_set, n_bootstraps = 500):

        """
        Method that calculates the point estimate NRMSE for the predictions from a global model of a certain type, using either the training or test set, and calculates a 95% confidence interval for the NRMSE using bootstrapping.

        Parameters
        ----
        model_type: str
            model for which the NRMSE will be calculated (options: "rf", "xgb", "hrf", "hxgb")
        
        X_combined, y_combined: dataframe, pandas Series
            training and validation data

        X_test, y_test: dataframe, pandas Series
            test data
        
        epsilon: float
            the small value that was added to the relative risk before log-transforming.

        which_set: str
            indicates which set to use for the NRMSE calculation and bootstrapping ("train" or "test")
        
        n_bootstraps: int
            number of bootstrap resamples to use for the confidence interval calculation
        
        Returns
        ----
            self.canton: str
                the canton included in the class instance

            point: float    
                the point estimate for the NRMSE, calculated by averaging the NRMSEs from the bootstrap resamples

            lower_bound: float
                the lower bound of the 95% confidence interval for the NRMSE, calculated by taking the 2.5th percentile of the NRMSEs from the bootstrap resamples

            upper_bound: float
                the upper bound of the 95% confidence interval for the NRMSE, calculated by taking the 97.5th percentile of the NRMSEs from the bootstrap resamples
        """

        start = time.perf_counter()

        try:
            grid_classi = joblib.load(f'../models/saved_models/{model_type}_classi_full.joblib')
            modelo_classi = grid_classi.best_estimator_
        except FileNotFoundError: 
            grid_classi = None
            modelo_classi = None

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
        try:
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_reg_full.joblib')
        except FileNotFoundError: 
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_full.joblib')
            
        modelo_reg = grid_reg.best_estimator_

        expected_features = list(modelo_reg.feature_names_in_)
    
        urb_features = [col for col in expected_features if col.startswith("urb_")]
            
        for X in [X_combined, X_test]:
            for col in urb_features:
                if col not in X.columns:
                    X[col] = 0
     
        X_combined = X_combined[expected_features]
        X_test = X_test[expected_features]
    
        if which_set == "test":
            bs = IIDBootstrap(
                X_test,
                y_test,
                seed = 42
            )
            n = len(X_test)
        elif which_set == "train":
            bs = IIDBootstrap(
                X_combined,
                y_combined,
                seed = 42
            )
            n = len(X_combined)
        else:
            print("Choose valid set")
    
        nrmse = []
    
        if grid_classi == None and model_type == "hrf":
            print("Only one class present in train set for classification. Skipping classification step.")
        elif grid_classi == None and model_type == "hxgb":
            print("Only one class present in train set for classification. Skipping classification step.")

        if model_type == "hrf":
            thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hrf.csv")
            thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == "full"]["best_th"].values[0]
        elif model_type == "hxgb":
            thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hxgb.csv")
            thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == "full"]["best_th"].values[0]
    
        for data in bs.bootstrap(n_bootstraps):
    
            X_boot = data[0][0]
            y_boot = data[0][1]
    
            if model_type == "rf" or model_type == "xgb" or grid_classi == None:
                preds = modelo_reg.predict(X_boot)
                preds = np.exp(preds) - epsilon
            elif model_type == "hrf" or model_type == "hxgb":
                pred_reg = modelo_reg.predict(X_boot)
                pred_reg = np.exp(pred_reg) - epsilon
                prob_classi = modelo_classi.predict_proba(X_boot)[:, 1]
                preds = np.where(prob_classi < best_threshold, 0, pred_reg)

            y_true = np.exp(y_boot) - epsilon
            nrmse.append(root_mean_squared_error(y_true = y_true, y_pred = preds) / (np.mean(y_true) + epsilon))

        point = np.mean(nrmse)
        lower_bound = np.percentile(nrmse, 2.5)
        upper_bound = np.percentile(nrmse, 97.5)

        end = time.perf_counter()
        
        print(f"Elapsed time: {end - start:.2f} seconds")
    
        return self.canton, point, lower_bound, upper_bound
