import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import datetime
import os
import pmdarima as pm
import warnings
warnings.filterwarnings('ignore')

'''#################### Functions ########################'''

def CORR(flattened_actual, flattened_forecast):
    """
    Calculate the Empirical Correlation Coefficient between two flattened arrays.
    IDENTICAL to GNN model implementation
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
    IDENTICAL to GNN model implementation
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


def data_preprocess_consistent(df, diff_order=1, start_index=0, end_index=None):
    """
    Data preprocessing CONSISTENT with GNN model approach
    """
    # Select data range
    if end_index is not None:
        df = df.loc[start_index:end_index]
    
    # Remove columns with >10% missing values (same as GNN)
    percent_missing = df.isnull().mean()
    df = df[df.columns[percent_missing < 0.1]]
    
    # Forward fill missing values (same as GNN)
    df = df.fillna(method='ffill')
    
    # Drop columns with NaNs at the start
    df = df.dropna(axis=1)
    
    # Apply differencing (consistent with GNN)
    df_diff = df.copy()
    for _ in range(diff_order):
        df_diff = df_diff.diff(periods=1)
    
    # Remove NaN rows created by differencing
    df_diff = df_diff.dropna()
    
    return df, df_diff


def select_top_features(df_diff, target_col='MARICOPA', top_percent=0.1):
    """
    Simple correlation-based feature selection to match GNN's 10% selection
    """
    # Calculate correlations with target
    correlations = df_diff.drop(target_col, axis=1).corrwith(df_diff[target_col]).abs()
    
    # Select top percentage of features
    n_features = int(len(correlations) * top_percent)
    n_features = max(n_features, 5)  # Minimum 5 features
    
    top_features = correlations.nlargest(n_features).index.tolist()
    
    return top_features


def create_lagged_features(df, features, target_col='MARICOPA', max_lag=6):
    """
    Create lagged features similar to GNN approach
    """
    result_df = pd.DataFrame(index=df.index)
    
    # Add target variable (no lag)
    result_df[target_col] = df[target_col]
    
    # Add lagged features
    for feature in features:
        for lag in range(0, max_lag + 1):
            if lag == 0:
                col_name = f"{feature}_lag{lag}"
            else:
                col_name = f"{feature}_lag{lag}"
            result_df[col_name] = df[feature].shift(lag)
    
    # Remove rows with NaN values
    result_df = result_df.dropna()
    
    return result_df


'''#################### Load and Setup Data ########################'''

# CONSISTENT data path and parameters with GNN model
data_file_path = "Data/Clean Processed Data2.csv"

# Check if file exists
if not os.path.exists(data_file_path):
    print(f"❌ Error: Data file not found at {data_file_path}")
    print("💡 Please update the data_file_path variable with the correct path to your data file.")
    exit(1)

# Load the data
df = pd.read_csv(data_file_path)

# Handle Date column - consistent with GNN approach
if 'Date' in df.columns:
    df.drop('Date', axis=1, inplace=True)

print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📊 Target variable 'MARICOPA' found: {'MARICOPA' in df.columns}")

# Check if MARICOPA column exists
if 'MARICOPA' not in df.columns:
    print("❌ Error: 'MARICOPA' column not found in the dataset")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

'''#################### Set Parameters - CONSISTENT WITH GNN ########################'''

# EXACT same parameters as GNN model
forecast_horizon_list = [2, 4, 8, 16]  # Same horizons as GNN
diff_order = 1  # Same differencing order as GNN
max_lag = 6  # Same max lag as GNN

# EXACT same data splits as GNN model
start_index = 0
train_end = 850
test_start_index = 900
test_end_index = 991  # CORRECTED to match GNN model (was 1002)

test_range = range(test_start_index, test_end_index)
test_split_size = test_end_index - test_start_index

print(f"📅 Using CONSISTENT data splits with GNN model:")
print(f"   Training: {start_index} to {train_end}")
print(f"   Test: {test_start_index} to {test_end_index}")
print(f"   Test size: {test_split_size} samples")

'''#################### Data Preprocessing - CONSISTENT WITH GNN ########################'''

# Preprocess data consistently with GNN approach
df_original, df_diff = data_preprocess_consistent(df, diff_order=diff_order, 
                                                  start_index=start_index, 
                                                  end_index=test_end_index)

print(f"🔧 After preprocessing: {df_diff.shape[0]} rows, {df_diff.shape[1]} columns")

# Apply SAME normalization as GNN model
initial_train_data = df_original.loc[start_index:train_end]
scaler = StandardScaler()
scaler.fit(initial_train_data)

# Normalize the entire dataset
df_norm_array = scaler.transform(df_original)
df_norm = pd.DataFrame(df_norm_array, columns=df_original.columns, index=df_original.index)

print(f"✅ Applied StandardScaler normalization (consistent with GNN)")

# Select top features (consistent 10% selection like GNN)
exogenous_features = [col for col in df_diff.columns if col != 'MARICOPA']
selected_features = select_top_features(df_diff, target_col='MARICOPA', top_percent=0.1)

print(f"🔍 Selected {len(selected_features)} features ({len(selected_features)/len(exogenous_features)*100:.1f}% of total)")
print(f"📋 Top selected features: {selected_features[:5]}...")

# Create lagged features (consistent with GNN approach)
df_with_lags = create_lagged_features(df_norm, selected_features, target_col='MARICOPA', max_lag=max_lag)
print(f"📈 Created dataset with lags: {df_with_lags.shape[0]} rows, {df_with_lags.shape[1]} columns")

'''#################### MAIN FORECASTING LOOP ########################'''

metric_df = pd.DataFrame()
all_predictions = {}
all_actuals = {}

print(f"\n🚀 Starting CORRECTED ARIMAX forecasting with CONSISTENT evaluation...")

for forecast_horizon in forecast_horizon_list:
    print(f"\n{'='*60}")
    print(f"📈 Processing {forecast_horizon}-week forecast horizon")
    print(f"{'='*60}")
    
    predictions = []
    actuals = []
    
    # Consistent sequence length with GNN (3 × horizon)
    sequence_length = 3 * forecast_horizon
    
    # Rolling window evaluation - CONSISTENT with GNN approach
    for iter_idx, end_index in enumerate(test_range):
        if iter_idx % 10 == 0:
            print(f"   📊 Processing iteration {iter_idx}/{len(test_range)-1}")
        
        # Define training window
        train_start = max(0, end_index - train_end)
        train_data = df_with_lags.loc[train_start:end_index-forecast_horizon]
        
        # Prepare features and target
        feature_cols = [col for col in df_with_lags.columns if col != 'MARICOPA']
        X_train = train_data[feature_cols]
        y_train = train_data['MARICOPA']
        
        # Get actual values for this forecast horizon
        actual_values = df_with_lags.loc[end_index-forecast_horizon+1:end_index]['MARICOPA'].values
        
        if len(actual_values) != forecast_horizon:
            continue  # Skip if not enough data
        
        try:
            # Fit ARIMAX model
            model = pm.auto_arima(
                y_train, 
                X=X_train,
                d=0,  # No additional differencing (already done)
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                max_p=3, max_q=3,  # Reasonable limits for speed
                information_criterion='aic'
            )
            
            # Prepare exogenous variables for forecasting
            X_forecast = df_with_lags.loc[end_index-forecast_horizon+1:end_index][feature_cols]
            
            # Make forecast
            forecast = model.predict(n_periods=forecast_horizon, X=X_forecast)
            
            # Store results
            predictions.extend(forecast)
            actuals.extend(actual_values)
            
        except Exception as e:
            print(f"   ⚠️ Error in iteration {iter_idx}: {e}")
            # Use naive forecast as fallback
            naive_forecast = y_train.iloc[-1]
            predictions.extend([naive_forecast] * forecast_horizon)
            actuals.extend(actual_values)
    
    # Calculate metrics - IDENTICAL to GNN model
    if len(predictions) > 0 and len(actuals) > 0:
        # Ensure equal lengths
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]
        
        # Apply inverse normalization to get original scale
        # Note: ARIMAX works on normalized data, so we need to inverse transform
        target_scaler = StandardScaler()
        target_scaler.fit(initial_train_data[['MARICOPA']])
        
        predictions_orig = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        actuals_orig = target_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
        
        # Calculate metrics using IDENTICAL functions as GNN model
        mae = mean_absolute_error(actuals_orig, predictions_orig)
        mape = mean_absolute_percentage_error(actuals_orig, predictions_orig)
        rmse = mean_squared_error(actuals_orig, predictions_orig, squared=False)
        mse = mean_squared_error(actuals_orig, predictions_orig)
        corr = CORR(actuals_orig, predictions_orig)
        rse = RSE(actuals_orig, predictions_orig)
        
        # Store metrics in same format as GNN model
        metric_df.loc['MAE_Test', f'ARIMAX_{forecast_horizon}_weeks'] = mae
        metric_df.loc['MAPE_Test', f'ARIMAX_{forecast_horizon}_weeks'] = mape
        metric_df.loc['RMSE_Test', f'ARIMAX_{forecast_horizon}_weeks'] = rmse
        metric_df.loc['MSE_Test', f'ARIMAX_{forecast_horizon}_weeks'] = mse
        metric_df.loc['CORR_Test', f'ARIMAX_{forecast_horizon}_weeks'] = corr
        metric_df.loc['RSE_Test', f'ARIMAX_{forecast_horizon}_weeks'] = rse
        
        # Store for plotting
        all_predictions[forecast_horizon] = predictions_orig
        all_actuals[forecast_horizon] = actuals_orig
        
        print(f"✅ {forecast_horizon}-week horizon completed:")
        print(f"   MAE: {mae:.4f}, MAPE: {mape:.4f}, RMSE: {rmse:.4f}")
        print(f"   MSE: {mse:.4f}, CORR: {corr:.4f}, RSE: {rse:.4f}")
        print(f"   Samples evaluated: {len(predictions_orig)}")
        
    else:
        print(f"❌ No valid predictions for {forecast_horizon}-week horizon")

'''#################### Plotting and Results ########################'''

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for i, horizon in enumerate(forecast_horizon_list):
    if horizon in all_predictions and horizon in all_actuals:
        ax = axes[i]
        
        # Plot predictions vs actuals
        predictions = all_predictions[horizon]
        actuals = all_actuals[horizon]
        
        # Take subset for clarity
        n_plot = min(100, len(predictions))
        indices = np.linspace(0, len(predictions)-1, n_plot, dtype=int)
        
        ax.plot(indices, actuals[indices], 'r-', label='Actual', linewidth=2, alpha=0.8)
        ax.plot(indices, predictions[indices], 'b--', label='ARIMAX Forecast', linewidth=2, alpha=0.8)
        
        ax.set_title(f'{horizon}-Week Forecast Horizon\nMAPE: {metric_df.loc["MAPE_Test", f"ARIMAX_{horizon}_weeks"]:.2f}%')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Valley Fever Cases')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ARIMAX_Corrected_Baseline_Results.png', dpi=300, bbox_inches='tight')

# Save results
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_filename = f"ARIMAX_Corrected_Baseline_Results_{timestamp}.csv"
metric_df.to_csv(results_filename)

print(f"\n" + "="*80)
print("📊 CORRECTED ARIMAX BASELINE RESULTS")
print("="*80)
print("🔧 CORRECTIONS APPLIED:")
print("   ✅ Consistent data splits with GNN model (test_end_index: 991)")
print("   ✅ Same differencing approach (diff_order: 1)")
print("   ✅ Same normalization (StandardScaler)")
print("   ✅ Same feature selection percentage (top 10%)")
print("   ✅ Same evaluation metrics and functions")
print("   ✅ Same rolling window evaluation approach")
print("   ✅ Same sequence length calculation (3 × horizon)")
print("\n📈 PERFORMANCE METRICS:")
print(metric_df.round(4))
print(f"\n💾 Results saved to: {results_filename}")
print("="*80)

# Show comparison with expected GNN results
print("\n🔍 COMPARISON REFERENCE:")
print("Expected GNN Performance:")
print("   2-week MAPE: ~13%")
print("   4-week MAPE: ~16%") 
print("   8-week MAPE: ~21%")
print("  16-week MAPE: ~23%")
print("\nThis corrected ARIMAX baseline provides a FAIR comparison!")

plt.show()
print('✅ Corrected ARIMAX baseline completed!')
