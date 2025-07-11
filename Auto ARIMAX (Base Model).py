import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import  plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import statistics
import datetime
import os


'''#################### Functions ########################'''
def crosscorr(daten_df,product, minlag=0, maxlag =0):

    new_pearson = []
    lag_list    = []
    daten_df=daten_df.diff(periods=1).diff(periods=1)[2:]
    for column in daten_df.columns:
        d1 = daten_df[product]
        d2 = daten_df[column]
        rs = [d1.corr(d2.shift(x)) for x in range(minlag,maxlag+1)]
        offset = -(minlag+np.abs(rs).argmax())
        max_pearson = round(max(rs, key=abs), 2)
        new_pearson.append(max_pearson)
        lag_list.append(offset)
    return new_pearson, lag_list

def data_set_generation(timeseries_df, time_lag_df, time_lag_col, exogenous_variables_list):

    print('Generate data set...')
    # initialize data set with output variable (Glyphosate)
    data_set = timeseries_df.iloc[:][['Glyphosate USD']]
    #data_set.reset_index(inplace=True)
    forecast_dataset=pd.DataFrame()

    # list of input components
    input_comp_list = []
    # go row-wise through time lag data frame
    for col in timeseries_df.columns.tolist():
        if col in exogenous_variables_list:
            # obtain optimal time lag
            optimal_lag = int(time_lag_df.loc[col, time_lag_col])

            # generate new component name
            comp_new = f'{col} ({optimal_lag})'
            input_comp_list += [comp_new]

            # get start index based on optimal time lag
            time_series = timeseries_df.iloc[:optimal_lag][col]
            time_series.index = time_series.index + pd.DateOffset(weeks=-optimal_lag)
            forecast_time_series = timeseries_df.iloc[optimal_lag:][col][:forecast_horizon]

            # add to data set df
            data_set=pd.concat([data_set, time_series.rename(comp_new)], axis=1)
            forecast_dataset[comp_new] = forecast_time_series.values

    data_set.dropna(inplace=True)
    forecast_dataset.set_index(X_test[:forecast_horizon].index, inplace=True)
    return data_set,forecast_dataset,input_comp_list

def data_preprocess(df, start_date,end_date):

    # Select the related range of data
    df = df.loc[start_date:end_date]
    #Find out the percentage of missing values in each column in the given dataset
    percent_missing = df.isnull().mean()
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    df=df[df.columns[percent_missing < 0.1]]
    df=df.fillna(method='ffill')
    # Drop columns with Nans at the start
    df=df.dropna(axis=1)

    # select exogenous variables columns
    variables_list = list(df)
    exogenous_variables_list=variables_list[:-1]
    return df, exogenous_variables_list


'''#################### Load data ########################'''
# Note: Dates in the dataset are adjusted and sorted to have values at equally spaced points in time.
name = os.getcwd()+'\\Weekly_Masterdata_2008_byOct14_merged_with_MonthlyMasterdata.csv'
df = pd.read_csv(name,encoding = "ISO-8859-1")
# Select date index
df['Friday_ofWeek'] = pd.to_datetime(df['Friday_ofWeek'])
df.set_index(['Friday_ofWeek'], inplace=True)
# Drop EXTRA columns


df.drop([

'Glyphosate USD',
'Monthly Glyphosate Acid (95%) Price RMB/t',
'Monthly.Avg Glyphosate_price RMB/T',
'Yearly.Avg Glyphosate_price RMB/T',
'Weekly_TC_FOB_Price_RMBperT',
'Monthly_Exchange_rate_in_RMB/$',
'MonthlyAve_Exchange_Rate',
'Monthly IDAN Price RMB/t',
'Monthly DEA Price USD/t',
'Monthly Glycine Price RMB/t',
'New_Glyphosate_USD',
'Year', 'Month', 'Year_Week'], axis=1, inplace=True)


#Select target variable
df['Glyphosate USD']=df['Weekly_TC_FOB_Price_USDperT']
df.drop(['Weekly_TC_FOB_Price_USDperT'], axis=1, inplace=True)


'''#################### Set Parameters ####################'''

forecast_horizon_list=[4, 12, 16] # how far into the future we want to forecast
differencing=1 # differencing order to keep the time series stationary all the time according to Test set
exogenous_variables=True

consider_extra_lags=0
start_date = datetime.datetime(2010, 1, 8)
test_start_date=datetime.datetime(2018, 8, 5)
end_date = datetime.datetime(2021, 12, 31)
test_split=len(df.loc[test_start_date:end_date])
min_number_of_EVs=10

'''#################### data preprocess ########################'''
df, exogenous_variables_list = data_preprocess(df, start_date, end_date)

'''#################### Feature Selection 1st Approach ########################'''
'''# example of mutual information feature selection for numerical input data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression


# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_regression, k=10)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    list=fs.get_support()
    EVs_list=X_train.columns[list]
    # transform train input data
    X_train_fs = pd.DataFrame(fs.transform(X_train), columns=EVs_list)
    # transform test input data
    X_test_fs = pd.DataFrame(fs.transform(X_test), columns=EVs_list)
    return X_train_fs, X_test_fs, fs, EVs_list

# load the dataset
X = df.drop('Glyphosate USD', axis=1)
y = df['Glyphosate USD']
# split into train and test sets
X_train, X_test= X[:-test_split], X[-test_split:]
y_train, y_test = y[:-test_split], y[-test_split:]
# feature selection
X_train_fs, X_test_fs, fs , EVs_list= select_features(X_train, y_train, X_test)


# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
f3 = plt.figure()
ax3 = f3.add_subplot(111)
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
ax3.set_xticklabels(exogenous_variables_list, rotation=90)
plt.tight_layout()
ax3.set_xticks(np.arange(len(exogenous_variables_list)))
pyplot.show()

# update exogenous variable list
exogenous_variables_list=EVs_list.tolist()
# select target and exogenous variables columns
variables_list = exogenous_variables_list.copy()
variables_list.append('Glyphosate USD')
df=df[variables_list]
# print(EVs_list)
'''


'''#################### Feature Selection 2nd Approach ########################'''
def feature_selection(df,test_split,exogenous_variables_list,forecast_horizon):
    from sklearn.linear_model import LassoCV, Lasso, ElasticNet, Ridge
    from sklearn.feature_selection import SelectFromModel
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    # Shift data according to the forecast horizon
    time_series = df.iloc[:-forecast_horizon]
    time_series.index = time_series.index + pd.DateOffset(weeks=forecast_horizon)

    # load the dataset
    X = time_series.drop('Glyphosate USD', axis=1)
    y = time_series['Glyphosate USD']
    # split into train and test sets
    X_train, X_test= X[:-test_split], X[-test_split:]
    y_train, y_test = y[:-test_split], y[-test_split:]
    # print (X_train.shape)

    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Lasso
    def lasso_regression(X_train, y_train, alpha):

        lassoreg = make_pipeline(StandardScaler(with_mean=True, with_std=True), Lasso(alpha=alpha, max_iter=1e6))
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
    ind = ['alpha_%.2g' % alpha_lasso[i] for i in range(0, 10)]
    coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)
    for i in range(10):
        # Using whole data range (X, y) for Lasso feature selection
        coef_matrix_lasso.iloc[i,] = lasso_regression(X,y, alpha_lasso[i])
        print("alpha {} is completed".format(i))

    #Counting non-zero elements within each row to select best alpha
    rows = (coef_matrix_lasso != 0).sum(1)
    selected_alpha=rows[(rows >= min_number_of_EVs)].index[-1]
    selected_alpha_row=coef_matrix_lasso.loc[selected_alpha][2:]
    exogenous_variables_list=(selected_alpha_row[selected_alpha_row!=0].index).tolist()
    return exogenous_variables_list





'''#################### Feature Selection 3rd Approach ########################'''
'''from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# load the dataset
X = df.drop('Glyphosate USD', axis=1)
y = df['Glyphosate USD']
# split into train and test sets
X_train, X_test= X[:-test_split], X[-test_split:]
y_train, y_test = y[:-test_split], y[-test_split:]

search = SelectFromModel(RandomForestRegressor())
search.fit(X_train, y_train)
feature_idx = search.get_support()
exogenous_variables_list = np.array(exogenous_variables_list)[feature_idx].tolist()


# exogenous_variables_list=
# select target and exogenous variables columns
variables_list = exogenous_variables_list.copy()
variables_list.append('Glyphosate USD')
df=df[variables_list]
'''



#########################
# MAIN LOOP
#########################
metric_df = pd.DataFrame()
f1 = plt.figure()
ax1 = f1.add_subplot(111)
f2 = plt.figure()
ax2 = f2.add_subplot(111)
for forecast_horizon in forecast_horizon_list:
    prediction_list=pd.Series(dtype='float64')
    lower_limits_list=pd.Series(dtype='float64')
    upper_limits_list=pd.Series(dtype='float64')

    # feature_selection that depends on forcast horizon
    exogenous_variables_list = feature_selection(df, test_split, exogenous_variables_list,forecast_horizon)
    variables_list = exogenous_variables_list.copy()
    variables_list.append('Glyphosate USD')
    df = df[variables_list]

    for it in range(test_split-forecast_horizon+1):
        '''#################### Test and Train Split ##################'''
        X = df[:-test_split+it].copy()
        X_test = df[-test_split+it:].copy()
        if it==0:
            X_test_initial=X_test
            Glyphosate_initial = X['Glyphosate USD']

        '''################## Exogenous Variables Offset ####################'''
        if exogenous_variables == True:

            # date offset
            min_lag = forecast_horizon
            max_lag = forecast_horizon+consider_extra_lags
            new_pearson = crosscorr(X, 'Glyphosate USD', min_lag, max_lag)
            new_pearson_df= pd.DataFrame(data=new_pearson, columns=X.columns,
                                           index=[f"max PC in -{min_lag}--{max_lag}",
                                                  f"best lag in -{min_lag}--{max_lag}"])
            time_lag_df = new_pearson_df.transpose()
            # name of column denoting optimal time lag
            time_lag_col = f'best lag in -{min_lag}--{max_lag}'

            #dataset generation
            X_offset,X_forecast, exogenous_variables_list_updated = data_set_generation(X,
                                                                               time_lag_df,
                                                                               time_lag_col,
                                                                               exogenous_variables_list)
            # Create arrays for the features and the target: X, y
            Glyphosate = X_offset['Glyphosate USD']
        else:
            Glyphosate = X['Glyphosate USD']

        '''######################## Automation ##########################'''
        import pmdarima as pm

        #Create and fit model
        if exogenous_variables==True:
            fit_results=pm.auto_arima(Glyphosate, d=differencing, trace=False, exog=X_offset[exogenous_variables_list_updated])
            model = SARIMAX(Glyphosate, order=fit_results.order ,seasonal_order=fit_results.seasonal_order ,
                            exog=X_offset[exogenous_variables_list_updated] ,initialization = 'approximate_diffuse')

        else:
            fit_results=pm.auto_arima(Glyphosate, d=differencing, trace=False)
            model = SARIMAX(Glyphosate, order=fit_results.order ,seasonal_order=fit_results.seasonal_order)

        results = model.fit()


        '''##################### Out of Sample Forecast ####################'''
        if exogenous_variables==True:
            arima_forecast = results.get_forecast(steps=forecast_horizon , exog=X_forecast[exogenous_variables_list_updated])
        else:
            arima_forecast = results.get_forecast(steps=forecast_horizon )
        # Extract prediction mean
        mean_forecast = arima_forecast.predicted_mean
        # Get confidence intervals of predictions
        confidence_intervals = arima_forecast.conf_int()
        # Select lower and upper confidence limits
        lower_limits = confidence_intervals.loc[:,'lower Glyphosate USD']
        upper_limits = confidence_intervals.loc[:,'upper Glyphosate USD']
        # Save prediction
        prediction_list = pd.concat([prediction_list, mean_forecast.tail(1)])
        lower_limits_list = pd.concat([lower_limits_list, lower_limits.tail(1)])
        upper_limits_list = pd.concat([upper_limits_list, upper_limits.tail(1)])


    '''##################### Accuracy metrics  ########################'''
    metric_df.loc['MAE_Test', 'Auto ARIMA_{} weeks'.format(forecast_horizon)] = mean_absolute_error(
        X_test_initial['Glyphosate USD'][forecast_horizon - 1:], prediction_list)
    metric_df.loc['MAPE_Test', 'Auto ARIMA_{} weeks'.format(forecast_horizon)] = mean_absolute_percentage_error(
        np.array(X_test_initial['Glyphosate USD'][forecast_horizon - 1:]), np.array(prediction_list))
    metric_df.loc['RMSE_Test', 'Auto ARIMA_{} weeks'.format(forecast_horizon)] = mean_squared_error(
        X_test_initial['Glyphosate USD'][forecast_horizon - 1:], prediction_list, squared=False)


    '''##################### plot the error ####################'''

    prediction_list.index = X_test_initial['Glyphosate USD'][forecast_horizon - 1:].index
    lower_limits_list.index = X_test_initial['Glyphosate USD'][forecast_horizon - 1:].index
    upper_limits_list.index = X_test_initial['Glyphosate USD'][forecast_horizon - 1:].index
    ax1.plot(prediction_list.index, prediction_list - X_test_initial['Glyphosate USD'][forecast_horizon - 1:],marker='.', label='{} weeks Model'.format(forecast_horizon))


    '''##################### plot all Forecast ####################'''

    # plot your mean forecast
    ax2.plot(prediction_list.index, prediction_list, marker='.', label='{} weeks Forecast'.format(forecast_horizon))
    # shade the area between your confidence limits
    # plt.fill_between(lower_limits_list.index, lower_limits_list,
    #                   upper_limits_list)




# plot the data
ax2.plot(Glyphosate_initial.index, Glyphosate_initial, marker='.', label='Train Set')
ax2.plot(X_test_initial.index, X_test_initial['Glyphosate USD'], color='black', marker='.', label='Test Set')
# set labels, legends and show plot
ax2.set_xlabel('Date')
ax2.set_ylabel('Glyphosate Price_ USD')
ax2.legend()
ax1.set_xlabel('Date')
ax1.set_ylabel('Error')
ax1.legend()

print(metric_df)
plt.show()
print('Done')






