"""
This script contains the Model class used to train and evaluate the different models for relative risk. It includes methods for partitioning the data, training the models, calculating variable importance, selecting features with RFECV, calculating prediction intervals, and calculating confidence intervals for the NRMSE metric. The class is designed to be flexible and can be used for different cantons and model types (Random Forest, XGBoost, Hybrid Random Forest, Hybrid XGBoost).
"""

import pandas as pd
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from xgboost import XGBClassifier
import joblib
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from arch.bootstrap import CircularBlockBootstrap
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from sklearn.base import clone


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

    def serie_temp_canton(self, model_type, y_pred_point, lower_bound, upper_bound):

        """
        Method that plots the time series of actual and predicted relative risk values for the test set, including the confidence intervals for the predictions. It reads the results from the csv file where they were saved after making predictions with the trained model.

        Parameters
        ----
        model_type: str
            the type of model for which to plot the results (options: "rf", "xgb", "hrf", "hxgb")
        lower_bound: array-like
            array-like with the lower bound of the confidence intervals for the predictions, to be used as the lower limit shaded area around the predicted values in the plot.
        upper_bound: array-like
            array-like with the upper bound of the confidence intervals for the predictions, to be used as the upper limit shaded area around the predicted values in the plot.
        Returns
        ----
            Nothing, just shows the plot with the time series of actual and predicted relative risk values for the test set, including the confidence intervals for the predictions.
        """

        results = pd.read_csv(f'../../data/model_results/results_{model_type}_{self.canton}.csv')
        
        results['week_canton'] = results['week_canton'].str.extract(r'(\d{4}-\d+)').iloc[:, 0].tolist()

        fig, ax = plt.subplots()
        ax.plot(results['week_canton'], results["actual"], label='Real', marker='o', linestyle = "dashed", color = "blue")
        ax.plot(results['week_canton'], results["pred"], label='Predicted', marker='o', color = "red")
        ax.plot(results['week_canton'], y_pred_point, label='Conformal predictions', marker='o', color = "green")
        ax.fill_between(x = results['week_canton'], label = "CI (95%)", y1 = lower_bound, y2 = upper_bound, alpha = 0.2, color = "green")
        ax.legend(loc = "upper right")
        self.ticks_years_top(ax, 20)
        ax.set_xlabel('Week')
        ax.set_ylabel(f'Relative risk')

        ax.set_title(f"{model_type} for {self.canton}")
        plt.show()
    
    def var_importance(self, X_test, y_test, model_type, repeats, var = "RR"):

        """
        Method that calculates and plots the permutation importance of the features for the specified model type, using the test set. It reads the trained model from the saved models folder, calculates the permutation importance using the sklearn function, and then creates a horizontal bar plot with the importance scores for the top 10 features.

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
        var: str
            the name of the target variable for which to calculate the variable importance, to be used in the title of the plot. the default value is "RR" for relative risk.

        Returns
        ----
            imp_df: dataframe
                dataframe with the features, their importance scores, and their standard deviation, sorted by importance score in descending order.
        """

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
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

    def rfecv_selection(self, model_type, X_combined, y_combined, X_test, y_test, pds, epsilon, repeats, var = "RR"):

        """ 
        Method that performs Recursive Feature Elimination with Cross-Validation (RFECV) to select the most important features for the specified model type, using the combined train and validation set. It reads the trained model from the saved models folder, performs RFECV to select the optimal number of features, and then fits a new model with only the selected features. Finally, it calculates the NRMSE for the test set using the new model and plots the feature importance for the selected features.

        Parameters
        ----
        model_type: str 
            the type of model for which to perform RFECV and feature selection (options: "rf", "xgb", "hrf", "hxgb")

        X_combined: dataframe
            dataframe with the features for the combined train and validation set

        y_combined: pandas Series
            pandas Series with the target variable (relative risk) for the combined train and validation set, log-transformed and with epsilon added
        
        X_test: dataframe
            dataframe with the features for the test set 
        
        y_test: pandas Series
            pandas Series with the target variable (relative risk) for the test set, log-transformed and with epsilon added

        pds: PredefinedSplit
            the predefined split object that specifies which data points belong to train and which to validation, to be used for RFECV.

        epsilon: float
            number used to transform the target variable, that is used here to transform the mean in case that it's zero when calculating NRMSE
        
        repeats: int
            the number of times to permute a feature for calculating the permutation importance, to be used as the n_repeats parameter in the sklearn function permutation_importance when calculating the feature importance for the selected features after performing RFECV.

        var: str
            the name of the target variable for which to perform RFECV and feature selection, to be used in the title of the plots. the default value is "RR" for relative risk.

        Returns
        ----
            Nothing, but it prints the optimal number of features selected by RFECV, the names of the selected features, and the new NRMSE for the test set using only the selected features. It also shows a plot with the RFECV score vs number of features, including a vertical line indicating the optimal number of features, and a horizontal bar plot with the feature importance for the selected features.
        """

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
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

        selected_mask = rfecv.support_ # boolean mask of selected features
        selected_features = X_combined.columns[selected_mask].tolist()

        print(f'Optimal number of features for {var} and {model_type}: {rfecv.n_features_}')
        print(f'Selected features: {selected_features}')

        # The cv_results_ dictionary may not always include the "n_features" key, depending on the version of sklearn and the specific estimator used. If it is not present, we can create a range of feature numbers based on the minimum number of features to select and the length of the mean_test_score array. This is used for plotting the RFECV score vs number of features, as the x-axis needs to represent the number of features selected at each step of the RFECV process.
        if "n_features" in rfecv.cv_results_:
            n_features_range = rfecv.cv_results_["n_features"]
        else:
            n_features_range = range(
                rfecv.min_features_to_select,
                rfecv.min_features_to_select + len(rfecv.cv_results_["mean_test_score"]) # This is used because the number of features is the minimum + the number of steps taken after the minimum features to select up until the total features
            )

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(n_features_range, rfecv.cv_results_['mean_test_score'], linewidth=2, marker='o', markersize=4)
        ax.fill_between( # Create shaded area for the standard deviation around the mean test score, to visualize the variability of the scores across the different folds of the cross-validation for each number of features selected.
            n_features_range,
            rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'],
            rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'],
            alpha=0.2
        )
        ax.axvline(rfecv.n_features_, linestyle='--', linewidth=2,
                label=f'Optimal: {rfecv.n_features_} features') # Add vertical line to indicate the optimal number of features selected by RFECV, with a label showing the number of features.
        ax.set_title(f'RFECV score vs Number of Features {var} and {model_type}', pad=10)
        ax.set_xlabel('Number of Features Selected')
        ax.set_ylabel('Cross-Validated score')
        ax.legend()
        plt.tight_layout()
        plt.show()

        final_model = clone(model) # Clone the original model to create a new model that we will fit with only the selected features, to avoid modifying the original model that was trained with all features. 
        final_model.fit(X_combined[selected_features], y_combined)

        final_model_pred = final_model.predict(X_test[selected_features])

        y_test = np.exp(y_test) - epsilon
        final_model_pred = np.exp(final_model_pred) - epsilon
        
        rmse = root_mean_squared_error(y_test, final_model_pred)

        nrmse = rmse / (np.mean(y_test) + epsilon)

        print(f"New NRMSE with selected features: {nrmse:.4f}")
        
        importance_scores = permutation_importance(final_model, X_test[selected_features], y_test, n_repeats=repeats, random_state=42, scoring='neg_mean_squared_error', n_jobs = -1).importances_mean

        df_importance = pd.DataFrame({'Feature': selected_features, 'Importance': importance_scores})
        df_importance = df_importance.sort_values(by='Importance', ascending=True)

        df_importance.plot(kind='barh', x='Feature', y='Importance', legend=False,
        figsize=(8, max(4, 0.35 * len(df_importance))))
        plt.title(f"Feature Importance Selected by RFECV for {model_type} model of {var}")
        plt.show()

    def prediction_intervals(self, X_combined, y_combined, X_test, epsilon, model_type, repeats):

        """
        Method that calculates the prediction intervals for the specified model type using the MAPIE library, which implements conformal prediction methods. It reads the trained model from the saved models folder, fits a CrossConformalRegressor with the best estimator from the grid search, and then predicts the point estimates and prediction intervals for the test set. Finally, it exponentiates the predictions and intervals to transform them back to the original scale of relative risk, and returns the point predictions (the mean) along with the lower and upper bounds of the prediction intervals.

        Parameters
        ----
        X_combined: dataframe
            dataframe with the features for the combined train and validation set
        
        y_combined: pandas Series
            pandas Series with the target variable (relative risk) for the combined train and validation set, log-transformed and with epsilon added
        
        X_test: dataframe
            dataframe with the features for the test set
        
        epsilon: float
            the small value that was added to the relative risk before log-transforming it, to avoid issues with zero values. This value is subtracted from the predictions and intervals after exponentiating them, to transform them back to the original scale of relative risk.
        
        model_type: str
            the type of model for which to calculate the prediction intervals (options: "rf", "xgb", "hrf", "hxgb")
        
        repeats: int
            how many bootstrap resamplings will be conducted

        Returns
        ----
            y_pred_point: numpy.ndarray
                array-like with the point predictions (the mean) for the test set, exponentiated and with epsilon subtracted to transform them back to the original scale of relative risk.

            lower_bound: numpy.ndarray
                array-like with the lower bound of the prediction intervals for the test set, exponentiated and with epsilon subtracted to transform them back to the original scale of relative risk. 
            
            upper_bound: numpy.ndarray
                array-like with the upper bound of the prediction intervals for the test set, exponentiated and with epsilon subtracted to transform them back to the original scale of relative risk.
        """

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
        try:
            grid = joblib.load(f'../../models/saved_models/{model_type}_reg_{self.canton}.joblib')
        except: 
            grid = joblib.load(f'../../models/saved_models/{model_type}_{self.canton}.joblib')

        best_rf = grid.best_estimator_

        cv_mapiets = BlockBootstrap(
            n_resamplings=repeats, length = 12, overlapping=True, random_state=42
        )

        mapie_reg = TimeSeriesRegressor(
            best_rf, method="enbpi", cv=cv_mapiets, agg_function="mean", n_jobs=-1, random_state = 42
        )

        mapie_reg.fit(X_combined, y_combined)

        y_pred_cp, y_interval = mapie_reg.predict(X_test, ensemble = True, confidence_level = 0.95)

        lower_bound = np.exp(y_interval[:, 0, 0]) - epsilon
        upper_bound = np.exp(y_interval[:, 1, 0]) - epsilon

        y_pred_point = np.exp(y_pred_cp) - epsilon

        return y_pred_point, lower_bound, upper_bound

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
                return the best threshold so that i can be saved to calculate intervals
        """

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
                f"../../data/model_results/results_{model_type}_{self.canton}.csv",
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

                reg_model.fit(X_combined_1, y_combined_1)
            else:
                    split_indices_1 = np.zeros(len(X_combined_1))
                    split_indices_1[:n_train_filtered] = 0
                    split_indices_1[n_train_filtered:] = -1
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

            if n_val_filtered == 0:
                print("No positive validation observations after filtering. Skipping hyperparameter tuning.")
                reg_model.fit(X_combined_1, y_combined_1)
                
            else:
                split_indices_1 = np.zeros(len(X_combined_1))
                split_indices_1[:n_train_filtered] = 0
                split_indices_1[n_train_filtered:] = -1
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

        if grid_classi != None:
            best_classi = grid_classi.best_estimator_

            split_indices = pds.test_fold
            val_mask = split_indices == 0
            X_val = X_combined.loc[val_mask].reset_index(drop=True)
            y_val = y_combined.loc[val_mask].reset_index(drop=True)

            thresholds = [x / 100 for x in range(0, 101)]
            probabilities_val = best_classi.predict_proba(X_val)[:, 1]
            y_test_actual = np.exp(y_test) - epsilon
            val_pred_reg = grid_reg.predict(X_val)
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

        results.to_csv(f"../../data/model_results/results_{model_type}_{self.canton}.csv", index=False)

        # Only save classification model if the classification step was not skipped due to only one class being present in test set
        if grid_classi != None:
            joblib.dump(grid_classi, f'../../models/saved_models/{model_type}_classi_{self.canton}.joblib')

        joblib.dump(grid_reg, f'../../models/saved_models/{model_type}_reg_{self.canton}.joblib')

        print(f"{model_type} ready for {self.canton}")

        return best_threshold

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

        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        for i, tick in enumerate(ax.xaxis.get_major_ticks()):
            if i % n != 0:
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.gridline.set_visible(False) 

    def calculate_nrmse(self, model_type, epsilon):

        """
        Method that calculates the NRMSE for the predictions from a model of a certain type, using the class instance canton.

        Parameters
        ----
        model_type: str
            model for which the NRMSE will be calculated (options: "rf", "xgb", "hrf", "hxgb")

        Returns
        ----
            self.canton: str
                the canton included in the class instance

            nrmse: float
                the calculated NRMSE
        """
                
        results = pd.read_csv(f'../../data/model_results/results_{model_type}_{self.canton}.csv')

        rmse = root_mean_squared_error(results["actual"], results["pred"])

        nrmse = rmse / (np.mean(results["actual"]) + epsilon)

        return self.canton, nrmse

    def confint_nrmse(self, model_type, X_combined, y_combined, X_test, y_test, epsilon, n_bootstraps = 100):

        """
        Method that calculates the confidence intervals for the NRMSE using Circular Block Bootstrap  

        Parameters
        ----
        model_type: str
            the type of model for which to calculate the confidence intervals (options: "rf", "xgb", "hrf", "hxgb")

        X_combined: dataframe
            dataframe with the features for the combined train and validation set
        
        y_combined: pandas Series
            pandas Series with the target variable (relative risk) for the combined train and validation set, log-transformed and with epsilon added
        
        X_test: dataframe
            dataframe with the features for the test set
        
        y_test: pandas Series
            pandas Series with the target variable (relative risk) for the test set, log-transformed and with epsilon added
        
        epsilon: float
            the small value that was added to the relative risk before log-transforming it, to avoid issues with zero values. This value is subtracted from the predictions and intervals after exponentiating them, to transform them back to the original scale of relative risk.
        
        n_bootstraps: int
            how many bootstrap resamplings will be conducted

        Returns
        ----
            Nothing, but prints the point prediction and upper and lower bounds of the interval
        """

        # Check if there's a classifier model if it's a hybrid model type
        try:
            grid_classi = joblib.load(f'../../models/saved_models/{model_type}_classi_{self.canton}.joblib')
        except FileNotFoundError: 
            grid_classi = None

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
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
                y_boot_binary = pd.Series((y_boot_actual > 0).astype(int))

                # Skip classification step if classifier was not used in the original model
                if grid_classi != None:
                    # Using TunedThresholdClassifier returns a different object that does not have a best_params_ attribute, but the resulting model can be obtained directly
                    try:
                        best_params_classi = grid_classi.best_params_
                        modelo_classi = RandomForestClassifier(**best_params_classi, random_state = 42)
                    except AttributeError:
                        modelo_classi = grid_classi.estimator_
                    modelo_classi.fit(X_boot, y_boot_binary)
                    models_classi.append(modelo_classi)

                mask_pos_train = y_boot_actual > 0 
                X_boot_1 = X_boot[mask_pos_train]
                y_boot_1 = y_boot[mask_pos_train]
                modelo_reg = RandomForestRegressor(**best_params_reg, random_state = 42)
                modelo_reg.fit(X_boot_1, y_boot_1)
                models.append(modelo_reg)
            elif model_type == "hxgb":
                y_boot_actual = np.exp(y_boot) - epsilon
                y_boot_binary = pd.Series((y_boot_actual > 0).astype(int))

                # Skip classification step if classifier was not used in the original model
                if grid_classi != None:
                    # Using TunedThresholdClassifier returns a different object that does not have a best_params_ attribute, but the resulting model can be obtained directly
                    try:
                        best_params_classi = grid_classi.best_params_
                        modelo_classi = XGBClassifier(**best_params_classi, random_state = 42)
                    except AttributeError:
                        modelo_classi = grid_classi.estimator_
                    modelo_classi.fit(X_boot, y_boot_binary)
                    models_classi.append(modelo_classi)

                mask_pos_train = y_boot_actual > 0 
                X_boot_1 = X_boot[mask_pos_train]
                y_boot_1 = y_boot[mask_pos_train]
                modelo_reg = XGBRegressor(**best_params_reg, random_state = 42)
                modelo_reg.fit(X_boot_1, y_boot_1)
                models.append(modelo_reg)

        # If there's no classification model in the hybrid case, predictions can be obtained directly from the regression model. However, in the other case, predicted cases by the regression model are changed if the classifier model first predicted that they were zero
        if grid_classi == None:                     
            predictions = np.array([model.predict(X_test) for model in models])
            predictions = np.exp(predictions) - epsilon
        else: 
            predictions_classi = np.array([model.predict(X_test) for model in models_classi])
            predictions_reg = np.array([model.predict(X_test) for model in models])
            predictions_reg = np.exp(predictions_reg) - epsilon
            predictions = np.array([np.where(y_pred_classi == 0, 0, y_pred_reg) for y_pred_classi, y_pred_reg in zip(predictions_classi, predictions_reg)])


        nrmse = np.array([root_mean_squared_error(y_true = y_test, y_pred = pred) / (np.mean(y_test) + epsilon) for pred in predictions])

        y_pred_point = np.mean(nrmse)

        lower_bound = np.percentile(nrmse, 2.5)
        upper_bound = np.percentile(nrmse, 97.5)

        print(f"point estimate: {round(y_pred_point, 2)}")
        print(f"lower bound: {round(lower_bound, 2)}, upper bound: {round(upper_bound, 2)}")