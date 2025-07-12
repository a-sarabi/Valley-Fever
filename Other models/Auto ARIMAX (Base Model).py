import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import statistics
import datetime
import os

'''#################### Functions ########################'''


def crosscorr(daten_df, product, minlag=0, maxlag=0):
    """
    Calculate cross-correlation between target variable and all other variables
    """
    new_pearson = []
    lag_list = []
    daten_df = daten_df.diff(periods=1).diff(periods=1)[2:]

    for column in daten_df.columns:
        if column != product:  # Skip self-correlation
            d1 = daten_df[product]
            d2 = daten_df[column]
            rs = [d1.corr(d2.shift(x)) for x in range(minlag, maxlag + 1)]
            offset = -(minlag + np.abs(rs).argmax())
            max_pearson = round(max(rs, key=abs), 2)
            new_pearson.append(max_pearson)
            lag_list.append(offset)
        else:
            new_pearson.append(1.0)  # Perfect correlation with itself
            lag_list.append(0)

    return new_pearson, lag_list


def data_set_generation(timeseries_df, time_lag_df, time_lag_col, exogenous_variables_list):
    """
    Generate dataset with optimal time lags for exogenous variables
    """
    print('Generate data set...')

    # Initialize dataset with target variable (MARICOPA)
    data_set = timeseries_df.iloc[:][['MARICOPA']]
    forecast_dataset = pd.DataFrame()

    # List of input components
    input_comp_list = []

    # Go through each exogenous variable
    for col in exogenous_variables_list:
        if col in timeseries_df.columns and col != 'MARICOPA':
            # Obtain optimal time lag
            optimal_lag = int(time_lag_df.loc[col, time_lag_col])

            # Generate new component name
            comp_new = f'{col} ({optimal_lag})'
            input_comp_list += [comp_new]

            # Get time series with optimal lag
            time_series = timeseries_df.iloc[:optimal_lag][col]
            time_series.index = time_series.index + pd.DateOffset(weeks=-optimal_lag)
            forecast_time_series = timeseries_df.iloc[optimal_lag:][col][:forecast_horizon]

            # Add to dataset
            data_set = pd.concat([data_set, time_series.rename(comp_new)], axis=1)
            forecast_dataset[comp_new] = forecast_time_series.values

    data_set.dropna(inplace=True)
    if len(forecast_dataset) > 0:
        forecast_dataset.set_index(X_test[:forecast_horizon].index, inplace=True)

    return data_set, forecast_dataset, input_comp_list


def data_preprocess(df, start_date, end_date):
    """
    Preprocess the Valley Fever dataset
    """
    # Select the related range of data
    if start_date is not None and end_date is not None:
        df = df.loc[start_date:end_date]

    # Find percentage of missing values in each column
    percent_missing = df.isnull().mean()
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})

    # Keep columns with less than 10% missing values
    df = df[df.columns[percent_missing < 0.1]]

    # Forward fill missing values
    df = df.fillna(method='ffill')

    # Drop columns with NaNs at the start
    df = df.dropna(axis=1)

    # Select exogenous variables columns (all except target)
    variables_list = list(df.columns)
    exogenous_variables_list = [col for col in variables_list if col != 'MARICOPA']

    return df, exogenous_variables_list


def feature_selection(df, test_split, exogenous_variables_list, forecast_horizon, min_number_of_EVs=10):
    """
    Feature selection using Lasso regression adapted for Valley Fever data
    """
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.feature_selection import SelectFromModel
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # Shift data according to the forecast horizon
    time_series = df.iloc[:-forecast_horizon]
    time_series.index = time_series.index + pd.DateOffset(weeks=forecast_horizon)

    # Prepare data
    X = time_series.drop('MARICOPA', axis=1)
    y = time_series['MARICOPA']

    # Split into train and test sets
    X_train, X_test = X[:-test_split], X[-test_split:]
    y_train, y_test = y[:-test_split], y[-test_split:]

    def lasso_regression(X_train, y_train, alpha):
        lassoreg = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                                 Lasso(alpha=alpha, max_iter=1e6))
        # Fit the model
        lassoreg.fit(X_train, y_train)
        y_pred = lassoreg.predict(X_train)
        # Return the result in pre-defined format
        rss = sum((y_pred - y_train) ** 2)
        ret = [rss]
        ret.extend([lassoreg[1].intercept_])
        ret.extend(lassoreg[1].coef_)
        return ret

    # Define the alpha values to test
    alpha_lasso = [1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 2, 5, 10, 20]

    # Initialize the dataframe to store coefficients
    col = ['rss', 'intercept'] + [i for i in exogenous_variables_list]
    ind = ['alpha_%.2g' % alpha_lasso[i] for i in range(0, len(alpha_lasso))]
    coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

    for i in range(len(alpha_lasso)):
        try:
            # Using whole data range (X, y) for Lasso feature selection
            coef_matrix_lasso.iloc[i,] = lasso_regression(X, y, alpha_lasso[i])
            print(f"alpha {i} is completed")
        except Exception as e:
            print(f"Error with alpha {i}: {e}")
            continue

    # Counting non-zero elements within each row to select best alpha
    rows = (coef_matrix_lasso != 0).sum(1)
    valid_rows = rows[rows >= min_number_of_EVs]

    if len(valid_rows) > 0:
        selected_alpha = valid_rows.index[-1]
        selected_alpha_row = coef_matrix_lasso.loc[selected_alpha][2:]
        selected_features = (selected_alpha_row[selected_alpha_row != 0].index).tolist()
        return selected_features
    else:
        # Fallback: return top features by absolute correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        return correlations.head(min_number_of_EVs).index.tolist()


def CORR(flattened_actual, flattened_forecast):
    """
    Calculate the Empirical Correlation Coefficient between two flattened arrays.
    """
    flattened_actual = np.array(flattened_actual)
    flattened_forecast = np.array(flattened_forecast)

    sigma_p = flattened_forecast.std()
    sigma_g = flattened_actual.std()
    mean_p = flattened_forecast.mean()
    mean_g = flattened_actual.mean()

    if sigma_g != 0 and sigma_p != 0:
        correlation = np.mean(
            ((flattened_forecast - mean_p) * (flattened_actual - mean_g)) / (sigma_p * sigma_g)
        )
    else:
        correlation = 0

    return correlation


def RSE(flattened_actual, flattened_forecast):
    """
    Calculate the Root Relative Squared Error between two flattened arrays.
    """
    flattened_actual = np.array(flattened_actual)
    flattened_forecast = np.array(flattened_forecast)

    mean_actual = np.mean(flattened_actual)
    numerator = np.sum((flattened_actual - flattened_forecast) ** 2)
    denominator = np.sum((flattened_actual - mean_actual) ** 2)

    if denominator != 0:
        rse = np.sqrt(numerator / denominator)
    else:
        rse = np.nan

    return rse


'''#################### Load data ########################'''
# Path to your Valley Fever data file - UPDATE THIS PATH
data_file_path = "D:/Shared/desktop 3/Valley Fever/Weather Data MARICOPA (MMWR Weeks)/Clean Data(Maricopa)/Clean Processed Data2.csv"

# Check if file exists
if not os.path.exists(data_file_path):
    print(f"❌ Error: Data file not found at {data_file_path}")
    print("💡 Please update the data_file_path variable with the correct path to your data file.")
    exit(1)

# Load the data
df = pd.read_csv(data_file_path)

# Handle date column if it exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
elif df.index.dtype == 'object':
    # Try to convert index to datetime if it's not already
    try:
        df.index = pd.to_datetime(df.index)
    except:
        print("Warning: Could not convert index to datetime. Using integer index.")

print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📊 Target variable 'MARICOPA' found: {'MARICOPA' in df.columns}")

# Check if MARICOPA column exists
if 'MARICOPA' not in df.columns:
    print("❌ Error: 'MARICOPA' column not found in the dataset")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

'''#################### Set Parameters ####################'''
# Match the forecast horizons from the GNN model
forecast_horizon_list = [2, 4, 8, 16]  # 2, 4, 8, 16 weeks horizon from GNN model

differencing = 1  # differencing order to keep the time series stationary
exogenous_variables = True
consider_extra_lags = 0

# Date ranges - adjust these based on your data
start_index = 0
train_end = 850
test_start_index = 900
test_end_index = 1002

# Convert indices to actual data range if using datetime index
if isinstance(df.index, pd.DatetimeIndex):
    start_date = df.index[start_index] if start_index < len(df) else df.index[0]
    end_date = df.index[test_end_index - 1] if test_end_index <= len(df) else df.index[-1]
    test_start_date = df.index[test_start_index] if test_start_index < len(df) else df.index[-100]
else:
    start_date = None
    end_date = None
    test_start_date = None

test_split = test_end_index - test_start_index  # Length of test set
min_number_of_EVs = 10

print(f"📅 Data range: {start_date} to {end_date}")
print(f"🧪 Test split size: {test_split}")

'''#################### Data preprocessing ########################'''
df_processed, exogenous_variables_list = data_preprocess(df, start_date, end_date)
print(f"🔧 After preprocessing: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")
print(f"📋 Number of exogenous variables: {len(exogenous_variables_list)}")

#########################
# MAIN LOOP
#########################
metric_df = pd.DataFrame()
f1 = plt.figure(figsize=(15, 10))
ax1 = f1.add_subplot(111)
f2 = plt.figure(figsize=(15, 10))
ax2 = f2.add_subplot(111)

print("\n🚀 Starting ARIMA forecasting for all horizons...")

for forecast_horizon in forecast_horizon_list:
    print(f"\n📈 Processing {forecast_horizon}-week forecast horizon...")

    prediction_list = pd.Series(dtype='float64')
    lower_limits_list = pd.Series(dtype='float64')
    upper_limits_list = pd.Series(dtype='float64')

    # Feature selection that depends on forecast horizon
    try:
        selected_features = feature_selection(df_processed, test_split,
                                              exogenous_variables_list, forecast_horizon)
        print(f"🔍 Selected {len(selected_features)} features for {forecast_horizon}-week horizon")

        # Update the dataframe to include only selected features + target
        variables_list = selected_features.copy()
        variables_list.append('MARICOPA')
        df_subset = df_processed[variables_list].copy()
    except Exception as e:
        print(f"⚠️ Feature selection failed for {forecast_horizon}-week horizon: {e}")
        print("Using all available features...")
        df_subset = df_processed.copy()
        selected_features = exogenous_variables_list.copy()

    for it in range(test_split - forecast_horizon + 1):
        if it % 10 == 0:
            print(f"   📊 Processing iteration {it}/{test_split - forecast_horizon}")

        '''#################### Test and Train Split ##################'''
        X = df_subset[:-test_split + it].copy()
        X_test = df_subset[-test_split + it:].copy()

        if it == 0:
            X_test_initial = X_test
            MARICOPA_initial = X['MARICOPA']

        '''################## Exogenous Variables Offset ####################'''
        if exogenous_variables == True and len(selected_features) > 0:
            # Calculate cross-correlation for optimal lags
            min_lag = forecast_horizon
            max_lag = forecast_horizon + consider_extra_lags

            try:
                new_pearson = crosscorr(X, 'MARICOPA', min_lag, max_lag)
                new_pearson_df = pd.DataFrame(data=new_pearson, columns=X.columns,
                                              index=[f"max PC in -{min_lag}--{max_lag}",
                                                     f"best lag in -{min_lag}--{max_lag}"])
                time_lag_df = new_pearson_df.transpose()
                time_lag_col = f'best lag in -{min_lag}--{max_lag}'

                # Dataset generation with optimal lags
                X_offset, X_forecast, exogenous_variables_list_updated = data_set_generation(
                    X, time_lag_df, time_lag_col, selected_features)

                MARICOPA = X_offset['MARICOPA']
            except Exception as e:
                print(f"   ⚠️ Error in lag calculation: {e}. Using original data.")
                MARICOPA = X['MARICOPA']
                X_offset = X
                X_forecast = pd.DataFrame()
                exogenous_variables_list_updated = []
        else:
            MARICOPA = X['MARICOPA']
            X_offset = X
            X_forecast = pd.DataFrame()
            exogenous_variables_list_updated = []

        '''######################## ARIMA Model ##########################'''
        import pmdarima as pm

        try:
            # Create and fit model
            if exogenous_variables == True and len(exogenous_variables_list_updated) > 0:
                # Use exogenous variables
                exog_data = X_offset[exogenous_variables_list_updated]
                fit_results = pm.auto_arima(MARICOPA, d=differencing, trace=False,
                                            exog=exog_data, seasonal=False,
                                            error_action='ignore', suppress_warnings=True)

                model = SARIMAX(MARICOPA,
                                order=fit_results.order,
                                seasonal_order=fit_results.seasonal_order,
                                exog=exog_data,
                                initialization='approximate_diffuse')
            else:
                # Use only endogenous variable
                fit_results = pm.auto_arima(MARICOPA, d=differencing, trace=False,
                                            seasonal=False, error_action='ignore',
                                            suppress_warnings=True)

                model = SARIMAX(MARICOPA,
                                order=fit_results.order,
                                seasonal_order=fit_results.seasonal_order)

            results = model.fit(disp=False)

            '''##################### Out of Sample Forecast ####################'''
            if (exogenous_variables == True and len(exogenous_variables_list_updated) > 0
                    and len(X_forecast) > 0):
                arima_forecast = results.get_forecast(steps=forecast_horizon,
                                                      exog=X_forecast[exogenous_variables_list_updated])
            else:
                arima_forecast = results.get_forecast(steps=forecast_horizon)

            # Extract prediction mean
            mean_forecast = arima_forecast.predicted_mean
            # Get confidence intervals of predictions
            confidence_intervals = arima_forecast.conf_int()
            # Select lower and upper confidence limits
            lower_limits = confidence_intervals.iloc[:, 0]
            upper_limits = confidence_intervals.iloc[:, 1]

            # Save prediction
            prediction_list = pd.concat([prediction_list, mean_forecast.tail(1)])
            lower_limits_list = pd.concat([lower_limits_list, lower_limits.tail(1)])
            upper_limits_list = pd.concat([upper_limits_list, upper_limits.tail(1)])

        except Exception as e:
            print(f"   ⚠️ Error in ARIMA fitting for iteration {it}: {e}")
            # Use simple naive forecast as fallback
            naive_forecast = MARICOPA.iloc[-1]
            prediction_list = pd.concat([prediction_list, pd.Series([naive_forecast])])
            lower_limits_list = pd.concat([lower_limits_list, pd.Series([naive_forecast * 0.9])])
            upper_limits_list = pd.concat([upper_limits_list, pd.Series([naive_forecast * 1.1])])

    '''##################### Accuracy metrics  ########################'''
    try:
        actual_values = X_test_initial['MARICOPA'][forecast_horizon - 1:]
        predicted_values = prediction_list

        # Calculate metrics using the same functions as GNN model
        mae = mean_absolute_error(actual_values, predicted_values)
        mape = mean_absolute_percentage_error(actual_values, predicted_values)
        rmse = mean_squared_error(actual_values, predicted_values, squared=False)
        mse = mean_squared_error(actual_values, predicted_values)
        corr = CORR(actual_values.values, predicted_values.values)
        rse = RSE(actual_values.values, predicted_values.values)

        # Store metrics
        metric_df.loc['MAE_Test', f'ARIMA_{forecast_horizon}_weeks'] = mae
        metric_df.loc['MAPE_Test', f'ARIMA_{forecast_horizon}_weeks'] = mape
        metric_df.loc['RMSE_Test', f'ARIMA_{forecast_horizon}_weeks'] = rmse
        metric_df.loc['MSE_Test', f'ARIMA_{forecast_horizon}_weeks'] = mse
        metric_df.loc['CORR_Test', f'ARIMA_{forecast_horizon}_weeks'] = corr
        metric_df.loc['RSE_Test', f'ARIMA_{forecast_horizon}_weeks'] = rse

        print(f"✅ {forecast_horizon}-week horizon completed:")
        print(f"   MAE: {mae:.4f}, MAPE: {mape:.4f}, RMSE: {rmse:.4f}")
        print(f"   MSE: {mse:.4f}, CORR: {corr:.4f}, RSE: {rse:.4f}")

    except Exception as e:
        print(f"❌ Error calculating metrics for {forecast_horizon}-week horizon: {e}")

    '''##################### Plotting ####################'''
    try:
        # Align indices for plotting
        prediction_list.index = X_test_initial['MARICOPA'][forecast_horizon - 1:].index
        lower_limits_list.index = X_test_initial['MARICOPA'][forecast_horizon - 1:].index
        upper_limits_list.index = X_test_initial['MARICOPA'][forecast_horizon - 1:].index

        # Plot prediction errors
        error = prediction_list - X_test_initial['MARICOPA'][forecast_horizon - 1:]
        ax1.plot(prediction_list.index, error, marker='.',
                 label=f'{forecast_horizon} weeks Model')

        # Plot forecasts
        ax2.plot(prediction_list.index, prediction_list, marker='.',
                 label=f'{forecast_horizon} weeks Forecast')

    except Exception as e:
        print(f"⚠️ Plotting error for {forecast_horizon}-week horizon: {e}")

# Plot the original data
try:
    ax2.plot(MARICOPA_initial.index, MARICOPA_initial, marker='.',
             label='Train Set', alpha=0.7)
    ax2.plot(X_test_initial.index, X_test_initial['MARICOPA'],
             color='black', marker='.', label='Test Set')
except Exception as e:
    print(f"⚠️ Error plotting original data: {e}")

# Set labels, legends and show plots
ax2.set_xlabel('Date')
ax2.set_ylabel('MARICOPA (Valley Fever Cases)')
ax2.set_title('ARIMA Forecasting Results - Valley Fever Cases')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax1.set_xlabel('Date')
ax1.set_ylabel('Prediction Error')
ax1.set_title('ARIMA Prediction Errors by Forecast Horizon')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Save results
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_filename = f"ARIMA_Valley_Fever_Results_{timestamp}.csv"
metric_df.to_csv(results_filename)

print("\n" + "=" * 60)
print("📊 FINAL RESULTS SUMMARY")
print("=" * 60)
print(metric_df)
print(f"\n💾 Results saved to: {results_filename}")
print("=" * 60)

plt.tight_layout()
plt.show()
print('✅ ARIMA forecasting completed!')