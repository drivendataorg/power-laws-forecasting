import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import sys
import os

from util import get_paths, prtime, update_params
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import psutil


GLOBAL_PARAMS_UPDATE = False            # If true, best parameters from the previous iteration are used in the next one
PREPROCESS_VERSION = 12                 # Version of preprocessed files
OUTPUT_VERSION = 1247                   # Version of output files to generate
IDIR, ODIR = get_paths()                # Folders with original and generated data
NUM_ATTEMPTS = 10                       # Number of attempts to optimize parameters and create better blended solution for each validation fold
RETRAIN = True                          # Retrain model on the whole training set (train+eval) after evaluation is done
VAL_SIZE = 0.3                          # Validation set ratio

def get_ratio(s1, s2):
    return (s1/s2).fillna(1).astype(np.float32)


#  Calculating static features, not using historical data

def get_static_features(df, metadata):
    df['Timestamp'] = pd.to_datetime(df.Timestamp)
    df['Doy'] = df.Timestamp.dt.dayofyear.astype(np.float32)
    df['Time'] = (df.Timestamp.dt.hour/24.0+df.Timestamp.dt.minute/(24.0*60.0)).astype(np.float32)
    df['DowTime'] = df['Dow']+df['Time']
    df['DoyTime'] = df['Doy']+df['Time']
    df['DowDoy'] = (df['Dow']+df['Doy']/1000.0).astype(np.float32)
    df['IsDayOffTime'] = df['IsDayOff'] + df['Time']
    df['IsDayOff2Time'] = df['IsDayOff2'] + df['Time']
    df['IsDayOffDoyTime'] = df['IsDayOff2']*1000 + df['Doy'] + df['Time']
    df['IsDayOff2DoyTime'] = df['IsDayOff2']*1000 + df['Doy'] + df['Time']


#   Metadata-based features
    for c in ['Surface', 'FridayIsDayOff','SaturdayIsDayOff','SundayIsDayOff']:
      df[c] = df['SiteId'].map(metadata[c])


#   One-hot encoding of SiteId
    for i in df['SiteId'].unique():
      name = 'SiteId_onehot_'+str(i)
      df[name] = (df['SiteId'] == i).astype(np.int8)

    return df


#  Ratios between aggregate features


def get_ratio_features(df):

    df['LastWeekByPrevWeek'] = get_ratio(df['LastWeekValues'],df['PrevWeekValues'])
    df['Last2WeeksAverage'] = ((df['LastWeekValues']+df['PrevWeekValues'])/2.0).fillna(df['LastWeekValues'])
    df['LastWeekByForecastMean'] = get_ratio(df['LastWeekValues'],df['MeanByForecastId'])
    df['LastWeekBySiteMean'] = get_ratio(df['LastWeekValues'],df['MeanBySiteId'])
    df['ForecastMeanBySiteMean'] = get_ratio(df['MeanByForecastId'],df['MeanBySiteId'])
    df['ForecastMedianBySiteMedian'] = get_ratio(df['MedianByForecastId'],df['MedianBySiteId'])
    df['LastWeekBySiteMax'] = get_ratio(df['LastWeekValues'], df['MaxBySiteId'])
    df['LastWeekBySiteMin'] = get_ratio(df['LastWeekValues'], df['MinBySiteId'])
    df['SiteMaxBySiteMin'] = get_ratio(df['MaxBySiteId'], df['MinBySiteId'])
    df['SiteMeanBySiteMedian'] = get_ratio(df['MeanBySiteId'], df['MedianBySiteId'])
    df['LastDayByPrev'] = get_ratio(df['LastDayValues'], df['MeanBySiteIdTime1D_offset1W'])
    df['LastHourByPrev'] = get_ratio(df['MeanBySiteId1h'], df['MeanBySiteId1h_offset1W'])
    df['CurTempMinusMonthMedian'] = (df['Temperature'] - df['MedianTemperatureBySiteIdTime1M']).fillna(0)

#    df['LastDayBySiteMean'] = get_ratio(df['LastDayValues'],df['MeanBySiteId10Y'])
#    df['LastHourBySiteMean'] = get_ratio(df['MeanBySiteId1h'],df['MeanBySiteId10Y'])



    return df    



  

#  NWRMSE loss function : slow version, works with variable lenghts of prediction periods


def nwrmse(y_true, y_pred, val_sub, freq):
  n_obs = 200  # longest forecast is ~192 obs
  weights = np.arange(1, n_obs + 1, dtype=np.float64)
  weights = (3 * n_obs - (2 * weights) + 1) / (2 * (n_obs ** 2))

  y_true = y_true.values
  y_pred[np.isnan(y_true)] = 0
  y_true[np.isnan(y_true)] = 0
#  print('nrmse : y_true = ', y_true)
#  print('nrmse : y_pred = ', y_pred)
  val_sub.reset_index(drop = True, inplace = True)
  forecast_ids = val_sub.ForecastId.unique()
  errs = []
  for i in forecast_ids:
    idx = val_sub.loc[val_sub.ForecastId == i].index
    true_mean = np.mean(y_true[idx])
    err = np.sqrt(np.sum(weights[:len(idx)]*((y_true[idx]-y_pred[idx])**2)))
    rel_err = err/(true_mean+1e-10)
#    print('forecast_id = ', i,' err = ', err,' rel_err = ', rel_err, 'len = ', len(idx), 'mean = ', true_mean)
#    print('y_true = ', y_true[idx])
#    print('val_sub = ', val_sub.loc[idx])
    errs.append(rel_err)
  return np.mean(errs)    


#  fast version - works on validation set with fixed forecast_len

def nwrmse_fast(y_true, y_pred, forecast_len):
  n_obs = 200  # longest forecast is ~192 obs
  weights = np.arange(1, n_obs + 1, dtype=np.float64)
  weights = (3 * n_obs - (2 * weights) + 1) / (2 * (n_obs ** 2))

  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  y_pred[np.isnan(y_true)] = 0
  y_true[np.isnan(y_true)] = 0
  errs = []
  for idx in range(0, len(y_true), forecast_len):
    err = (y_pred[idx:idx+forecast_len]-y_true[idx:idx+forecast_len])**2
    true_mean = np.mean(y_true[idx:idx+forecast_len])
    rel_err = np.sqrt(np.sum(err*weights[:len(err)]))/(true_mean+1e-10)
    errs.append(rel_err)
  return np.mean(errs)


#  NWRMSE metric for LightGBM


def lgb_nwrmse(forecast_len, my_log):
  def lgb_nwrmse_internal(y_pred, train_data):
    labels = np.array(train_data.get_label())
    y_pred = np.array(y_pred)
    if my_log:
      labels = np.expm1(labels)
      y_pred = np.expm1(y_pred)
    loss = nwrmse_fast(labels, y_pred, forecast_len)

#    print('errs len = ', len(errs), ' score = ', np.mean(errs))
    return 'lgb_nwrmse', loss, False
  return lgb_nwrmse_internal


#  Initial parameters for different sampling frequencies

best_lgb_params = {}

best_lgb_params[ 900000000000 ] =  {'use_HolidaysSomewhere': False, 'use_IsDayOffTime': True, 'use_ForecastId': True, 'min_gain_to_split': 0.001, 'use_IsDayOff': True, 'use_onehot': False, 'use_MeanByForecastId': True, 'use_ForecastMeanBySiteMean': False, 'bagging_freq': 1, 'use_MedianBySiteId': False, 'use_LastWeekValues': False, 'lambda_l1': 0.03, 'feature_fraction': 0.8, 'use_Time': False, 'use_IsDayOffDoyTime': True, 'use_SiteMaxBySiteMin': True, 'use_MinBySiteId': False, 'use_LastWeekBySiteMean': True, 'lambda_l2': 0, 'metric': 'lgb_nwrmse', 'use_MeanBySiteId': True, 'use_Last2WeeksAverage': False, 'use_obs_id': True, 'use_SiteMeanBySiteMedian': True, 'use_SiteId': True, 'min_sum_hessian_in_leaf': 0.001, 'use_ForecastMedianBySiteMedian': True, 'boosting_type': 'gbdt', 'use_LastWeekByPrevWeek': True, 'use_MinByForecastId': True, 'use_Dow': True, 'use_MaxBySiteId': True, 'use_IsHoliday': True, 'max_depth': 10, 'num_leaves': 2047, 'use_LastWeekByForecastMean': True, 'bagging_fraction': 0.8, 'use_LastWeekBySiteMin': False, 'use_DoyTime': True, 'use_LastWeekBySiteMax': True, 'use_Doy': False, 'objective': 'regression_l2', 'use_DowDoy': False, 'use_Temperature': True, 'use_LastDayValues': True, 'use_MedianByForecastId': True, 'use_MaxByForecastId': False, 'learning_rate': 0.1, 'my_keep_mean_nan': True, 'min_data_in_leaf': 1, 'max_bin': 1023, 'use_DowTime': True, 'use_PrevWeekValues': False, 'my_skip_first' : 0.0, 'my_log' : True, 'my_weights' : 'raw'}
best_lgb_params[ 3600000000000 ] =  {'use_HolidaysSomewhere': False, 'use_IsDayOffTime': True, 'use_ForecastId': False, 'min_gain_to_split': 0.001, 'use_IsDayOff': False, 'use_onehot': False, 'use_MeanByForecastId': False, 'use_ForecastMeanBySiteMean': True, 'bagging_freq': 2, 'use_MedianBySiteId': True, 'use_LastWeekValues': True, 'lambda_l1': 0.03, 'feature_fraction': 0.9, 'use_Time': True, 'use_IsDayOffDoyTime': True, 'use_SiteMaxBySiteMin': True, 'use_MinBySiteId': True, 'use_LastWeekBySiteMean': True, 'lambda_l2': 0, 'metric': 'lgb_nwrmse', 'use_MeanBySiteId': True, 'use_Last2WeeksAverage': True, 'use_obs_id': False, 'use_SiteMeanBySiteMedian': False, 'use_SiteId': True, 'min_sum_hessian_in_leaf': 0, 'use_ForecastMedianBySiteMedian': True, 'boosting_type': 'gbdt', 'use_LastWeekByPrevWeek': False, 'use_MinByForecastId': True, 'use_Dow': True, 'use_MaxBySiteId': True, 'use_IsHoliday': True, 'max_depth': 12, 'num_leaves': 2047, 'use_LastWeekByForecastMean': True, 'bagging_fraction': 0.8, 'use_LastWeekBySiteMin': True, 'use_DoyTime': True, 'use_LastWeekBySiteMax': True, 'use_Doy': True, 'objective': 'regression_l1', 'use_DowDoy': True, 'use_Temperature': True, 'use_LastDayValues': True, 'use_MedianByForecastId': False, 'use_MaxByForecastId': True, 'learning_rate': 0.1, 'my_keep_mean_nan': True, 'min_data_in_leaf': 10, 'max_bin': 2047, 'use_DowTime': False, 'use_PrevWeekValues': False, 'my_skip_first' : 0.0, 'my_log' : True, 'my_weights' : 'raw'}
best_lgb_params[ 86400000000000 ] =  {'use_ForecastId': False, 'use_ForecastMeanBySiteMean': False, 'max_bin': 2047, 'use_MaxBySiteIdTime10Y': False, 'use_Surface': True, 'lambda_l2': 100, 'use_MedianBySiteId1W': True, 'use_IsDayOff': False, 'use_MinBySiteId': True, 'use_LastWeekBySiteMin': True, 'use_MaxBySiteIdIsDayOffTime10Y': False, 'use_PrevWeekValues': False, 'use_IsHolidayTomorrow': True, 'use_MeanBySiteIdDow10Y': False, 'use_MaxBySiteId1W': True, 'objective': 'huber', 'use_MaxBySiteIdIsDayOff2Time10Y': True, 'use_DoyTime': True, 'use_MeanBySiteIdIsDayOff2Time10Y': False, 'use_MedianBySiteId': True, 'use_MedianBySiteIdIsDayOff3Time10Y': True, 'use_MeanBySiteIdTime10Y': True, 'use_Doy': True, 'use_SiteMaxBySiteMin': True, 'use_HolidaysSomewhereTomorrow': True, 'min_sum_hessian_in_leaf': 0, 'use_Temperature': True, 'use_MinBySiteIdIsDayOffTime10Y': True, 'use_MinBySiteId10Y': True, 'use_MinByForecastIdTime10Y': False, 'use_MedianBySiteIdDow10Y': True, 'use_MedianBySiteId1Y': True, 'my_skip_first': 0.0, 'use_dist_from_start': False, 'use_nancountBySiteId1D': True, 'use_LastWeekByForecastMean': True, 'use_IsDayOffTime': False, 'use_IsDayOff2Time': True, 'min_data_in_leaf': 20, 'use_LastDayByPrev': False, 'use_MinByForecastIdIsDayOffTime10Y': True, 'use_ForecastMedianBySiteMedian': True, 'use_IsDayOff2DoyTime': True, 'use_MeanBySiteId1h_offset1W': True, 'my_log': True, 'use_CurTempMinusMonthMedian': False, 'use_MaxByForecastId': True, 'use_MedianByForecastIdIsDayOffTime10Y': True, 'use_IsHolidayYesterday': True, 'use_MeanBySiteIdTime1D_offset1W': True, 'use_MedianBySiteIdTime10Y': True, 'use_LastDayValues': False, 'use_LastWeekByPrevWeek': True, 'bagging_freq': 1, 'use_LastWeekValues': True, 'use_MedianByForecastIdTime10Y': True, 'use_LastHourByPrev': True, 'use_MeanBySiteId1h': True, 'use_MinBySiteId1M': True, 'use_onehot': False, 'learning_rate': 0.05, 'use_MaxBySiteId1M': True, 'use_IsHoliday': True, 'use_DowDoy': False, 'use_MaxByForecastIdDow10Y': False, 'use_HolidaysSomewhereYesterday': True, 'use_MeanBySiteId1W': True, 'use_MinBySiteIdTime10Y': True, 'use_MeanByForecastIdDow10Y': True, 'use_MeanByForecastIdIsDayOffTime10Y': True, 'use_SaturdayIsDayOff': False, 'use_MeanBySiteId': False, 'use_Time': False, 'max_depth': 12, 'use_MaxBySiteIdDow10Y': True, 'use_IsDayOff2': False, 'use_obs_id': False, 'use_MaxBySiteId1Y': False, 'use_MedianByForecastIdDow10Y': False, 'use_MedianBySiteIdIsDayOff3DowTime10Y': False, 'use_MinBySiteIdDow10Y': True, 'use_MedianBySiteIdIsDayOff2DowTime10Y': True, 'use_SiteMeanBySiteMedian': False, 'use_nancountBySiteId10Y': True, 'use_MedianBySiteIdIsDayOffTime10Y': False, 'use_IsDayOff3': True, 'lambda_l1': 0.1, 'use_MaxByForecastIdTime10Y': False, 'use_SiteId': True, 'bagging_fraction': 0.8, 'use_Dow': False, 'use_MinBySiteId1W': True, 'use_MeanByForecastId': True, 'use_MedianBySiteIdIsDayOff2Time10Y_offset1W': True, 'use_MeanBySiteId1Y': False, 'use_MinByForecastId': False, 'my_weights': 'no', 'num_leaves': 2047, 'use_MeanBySiteId10Y': True, 'use_MedianTemperatureBySiteIdTime10Y': False, 'metric': 'lgb_nwrmse', 'boosting_type': 'gbdt', 'use_MeanBySiteId1M': False, 'use_MeanBySiteIdIsDayOffTime10Y': False, 'use_MedianBySiteIdIsDayOff2Time10Y': True, 'use_MeanByForecastIdTime10Y': True, 'use_nancountBySiteId1M': True, 'use_MaxBySiteId': False, 'use_LastWeekBySiteMax': False, 'use_LastWeekBySiteMean': True, 'min_gain_to_split': 0, 'use_nancountBySiteId1W': True, 'use_MedianTemperatureBySiteIdTime1M': True, 'use_MinBySiteIdIsDayOff2Time10Y': False, 'my_keep_mean_nan': True, 'use_MaxBySiteId10Y': False, 'use_HolidaysSomewhere': True, 'use_MedianByForecastId': False, 'use_MaxByForecastIdIsDayOffTime10Y': True, 'feature_fraction': 0.7, 'use_IsDayOffDoyTime': False, 'use_MedianBySiteId1M': True, 'use_MinByForecastIdDow10Y': False, 'use_lag': True, 'use_DowTime': True, 'use_MedianBySiteId10Y': False, 'use_MinBySiteId1Y': True, 'use_FridayIsDayOff': False, 'use_Last2WeeksAverage': False, 'use_SundayIsDayOff': True}




#  Creating train and validation sets


def split(df, random_seed, forecast_len, params):

  forecast_ids = df['ForecastId'].unique()
  train_ids, val_ids = train_test_split(forecast_ids, test_size = VAL_SIZE, random_state = random_seed)

  val_slices = []
  train_slices = []

  for i in val_ids:                                
    subset = df.loc[df.ForecastId == i]
    val_subset = subset.iloc[-forecast_len:]                # if this ForecastId is selected for validation - last part of its data goes to validation set
    val_slices.append(val_subset)

    train_start = int(params['my_skip_first']*len(subset))  
    train_subset = subset.iloc[train_start:-forecast_len]   # using remaining as train
    train_slices.append(train_subset)

  val_df = pd.concat(val_slices)

  for i in train_ids:
    subset = df.loc[df.ForecastId == i]
    train_start = int(params['my_skip_first']*len(subset))
    train_subset = subset.iloc[train_start:]   
    train_slices.append(train_subset)


#  train_sub = df.loc[df.ForecastId.isin(train_ids)]
  train_df = pd.concat(train_slices)

  for c in val_df.columns:
    if '_noval' in c:
      val_df[c[:-6]] = val_df[c]   # replacing values in validation set to exclude information not available in validation period
  val_df = get_ratio_features(val_df)
  
  return train_df, val_df     



#  Blending (averaging) new prediction with a list of preious ones. 
#  old : current list of predictions
#  new : this prediction replaces one of old predictions (at position rem_idx in the list) or just appended to the list if rem_idx == -1
#  use_log : when True, averaging is done on the log scale, otherwise linear


def blend(old, new, rem_idx, use_log):
  updated_list = old.copy()
  if rem_idx != -1:
    updated_list = updated_list[:rem_idx] + updated_list[rem_idx+1:]
  updated_list.append(new)
  blended = np.stack(updated_list)
  if use_log:
    blended = np.log1p(blended)
  blended = blended.mean(axis = 0)
  if use_log:
    blended = np.expm1(blended)
  return blended, updated_list



#  Model training and predicting happens here



def process(freq, params, best_loss, train, submission_format, blended_predictions, submission_frequency, random_seed = 14):
  

  forecast_lens = {900000000000 : 192, 3600000000000 : 192, 86400000000000 : 60}
  forecast_len = forecast_lens[freq]
  loss = {}
  gc.collect()
  forecast_ids = submission_frequency['ForecastId'].loc[submission_frequency.ForecastPeriodNS == freq]
  data_sub = train.loc[train.ForecastId.isin(forecast_ids)]
  submission_sub = submission_format.loc[submission_format.ForecastId.isin(forecast_ids)]


  n_obs = 200  # longest forecast is ~192 obs
  weights = np.arange(1, n_obs + 1, dtype=np.float64)
  weights = np.array((3 * n_obs - (2 * weights) + 1) / (2 * (n_obs ** 2)))
  if params['my_weights'] == 'norm':
    weights = weights/np.mean(weights[:forecast_len])  # normalizing weights

  print('data subset : len = ', len(data_sub))
  use_cols = []
  for p in params:
    if 'use_' in p and params[p]:
      if 'onehot' not in p:
        col_name = p[4:]
        if (col_name not in train.columns) or (col_name not in submission_format.columns):
          print('WARNING : ', col_name,'usage is turned ON in parameters, but it is absent in train columns')
        else:
          use_cols.append(col_name)
      else:
        onehot = [c for c in train.columns if 'onehot' in c]
        use_cols += onehot
  print('using columns : ', use_cols) 

  train_sub, val_sub = split(data_sub, random_seed = random_seed, forecast_len = forecast_len, params = params)  # custom split by forecast_id

  train_sub = train_sub.loc[np.isfinite(train_sub['Value'])]
  if params['my_keep_mean_nan'] == False:
    train_sub = train_sub.loc[np.isfinite(train_sub['MeanByForecastId'])]


  train_sub.iloc[0:10000].to_csv(ODIR + 'lgb_train_head10000.csv', index = False)
  val_sub.iloc[0:10000].to_csv(ODIR + 'lgb_val_head10000.csv', index = False)
  submission_sub[use_cols].iloc[0:10000].to_csv(ODIR+'lgb_test_head10000.csv', index = False)

  cat_idx = []
  for idx, c in enumerate(use_cols):
    if (c == 'SiteId') or (c == 'ForecastId'):
      cat_idx.append(idx)

  dtrain = lgb.Dataset(train_sub[use_cols].values, label = np.log1p(train_sub.Value) if params['my_log'] else train_sub.Value, 
        categorical_feature = cat_idx,  weight=weights[train_sub['lag'].astype(np.int)] if params['my_weights'] != 'no' else 1)
#  print(train_sub[use_cols])
  gc.collect()
  dval = lgb.Dataset(val_sub[use_cols].values, label = np.log1p(val_sub.Value) if params['my_log'] else val_sub.Value, 
    categorical_feature = cat_idx, reference = dtrain)
  print('mean val target = ', np.mean(val_sub.Value))
  print('mean train target = ', np.mean(train_sub.Value))
#  dtest = lgb.Dataset(submission_format[use_cols])
  lgb_params = {}
  for p in params:
    if ('use_' not in p) and ('my_' not in p):
      lgb_params[p] = params[p]
  mem_available = psutil.virtual_memory().total
  prtime('mem available = ', mem_available)
  if mem_available < 48*(2**30):
    print('using histogram_pool_size to lower memory usage')
    lgb_params['histogram_pool_size'] = 16384
  MAX_ROUNDS = 3000
  gc.collect()
  prtime('starting training')
  bst = lgb.train(
        lgb_params.copy(), dtrain, num_boost_round=MAX_ROUNDS, feval = lgb_nwrmse(forecast_len, params['my_log']),
        valid_sets=dval, early_stopping_rounds=50, verbose_eval=10, categorical_feature = cat_idx)
  prtime('training done')

  print("\n".join(("%s: %.2f" % x) for x in sorted(
    zip(train_sub[use_cols].columns, bst.feature_importance("gain")),
    key=lambda x: x[1], reverse=True  )))


  val_pred = bst.predict(val_sub[use_cols].values, num_iteration=bst.best_iteration or MAX_ROUNDS)
  val_target = val_sub.Value
  if params['my_log']:
#    pred = np.expm1(pred)
    val_pred = np.expm1(val_pred)
  print('mean val_pred = ', np.mean(val_pred), 'mean val_target = ', np.mean(val_target))
#  nwrmse_loss = nwrmse(np.expm1(val_sub.Value), val_pred, val_sub, freq)
  nwrmse_loss = nwrmse_fast(val_target, val_pred, forecast_len)
  print('nwrmse loss = ', nwrmse_loss)
  loss['single'] = nwrmse_loss
  sys.stdout.flush()

#  return loss
#  print('pred = ', pred)
#  print('val_pred = ', val_pred)
#  print('mean pred = ', np.mean(pred))
#  print('mean val_pred = ', np.mean(val_pred))

  best_blended_loss = 1e10
  best_blending_params = None
  best_blended_predictions = None
  for rem_idx in range(-1, len(blended_predictions['validation'])):
    for use_log in [False, True]:
      blended_val_pred, bp = blend(blended_predictions['validation'].copy(), val_pred, rem_idx, use_log)
#      bp = blended_predictions['validation'].copy()

      blended_loss = nwrmse_fast(val_target, blended_val_pred, forecast_len)      
      if blended_loss < best_blended_loss:
        best_blended_loss = blended_loss
        best_blending_params = {'rem_idx' : rem_idx, 'use_log' : use_log}
        best_blended_predictions = bp

  loss['blended'] = best_blended_loss
  print('best blended loss at this iteration = ', loss['blended'])
  sys.stdout.flush()

  if loss['blended'] < best_loss['blended']:
    print('new best blended loss', loss['blended'],' found' , 'with blending params = ', best_blending_params, 'predictions num =', len(best_blended_predictions))
#    blended_predictions['validation'].append(val_pred)     
    blended_predictions['validation'] = best_blended_predictions
    if RETRAIN:           # Retraining on the full training set, no validation
      prtime('retraining')
      data_sub = data_sub.loc[np.isfinite(data_sub['Value'])]   # Removing nans from train
      dtrain = lgb.Dataset(data_sub[use_cols].values, label = np.log1p(data_sub.Value) if params['my_log'] else data_sub.Value, 
        categorical_feature = cat_idx,  weight=weights[data_sub['lag'].astype(np.int)] if params['my_weights'] != 'no' else 1)
      gc.collect()
      bst = lgb.train(
        lgb_params.copy(), dtrain, num_boost_round=bst.best_iteration or MAX_ROUNDS, verbose_eval=10, categorical_feature = cat_idx)
    
    test_pred = bst.predict(submission_sub[use_cols].values, num_iteration=bst.best_iteration or MAX_ROUNDS)
    if params['my_log']:
      test_pred =np.expm1(test_pred)
    print('mean test_pred = ', np.mean(test_pred))
    
    blended_test_pred, blended_predictions['test'] = blend(blended_predictions['test'], test_pred, 
      best_blending_params['rem_idx'], best_blending_params['use_log'])
    print('mean blended_test_pred = ', np.mean(blended_test_pred))

    submission_format.loc[submission_format.ForecastId.isin(forecast_ids), 'Value'] = blended_test_pred
    val_sub['Predicted'] = blended_val_pred
    val_sub[['obs_id','SiteId','Timestamp','ForecastId','Value','Predicted']].to_csv(ODIR + 'validation_predicted_'+str(random_seed)+'_'+str(freq)+'.csv')
  return loss, submission_format


  
#  Trying different hyperparameters sets n_attempts times, creating blended prediction from best subset of hyperparameters sets


def tune_params(train, submission_format, submission_frequency, freq, n_attempts = 10, random_seed = 14):
  lgb_grid = {'boosting_type' : ['gbdt'],
  'learning_rate' : [0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2],
  'bagging_freq' : [1, 2, 3, 5, 10, 20, 50, 100],
  'feature_fraction' : [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6, 0.7,0.8,0.9,1.0],
  'bagging_fraction' : [0.2, 0.5, 0.7,0.8,0.85, 0.9, 0.95, 1.0],
  'max_depth' : [4,5,6,7,8,10,12,14,20],
  'num_leaves' : [7,15, 31, 63, 127, 255, 511, 1023, 2047],
  'min_data_in_leaf' : [1, 10, 20, 30, 50, 70, 100, 120, 150, 200, 300],
  'min_sum_hessian_in_leaf' : [0, 0.001, 0.01, 0.1, 1, 3, 10, 30, 100],
  'min_gain_to_split' : [0, 0.001, 0.01, 0.1, 1, 10, 100],
  'lambda_l1' : [0, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 10, 100],
  'lambda_l2' : [0, 0.001, 0.01, 0.1, 1, 10, 100],
  'max_bin' : [63, 127, 255, 511, 1023, 2047],
  'use_onehot' : [True, False],
  'my_keep_mean_nan' : [True],
  'my_skip_first' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  'my_log' : [True, False],
  'my_weights' : ['no', 'raw', 'norm'],
  'objective' : ['regression_l2','regression_l1','huber','fair'],
  'metric' : ['lgb_nwrmse']}

  global best_lgb_params

  best_params = best_lgb_params[freq].copy()
  for c in train.columns:
    if (c == 'Timestamp') or (c == 'Date') or (c == 'Value') or (c == 'Validation') or (c == 'original_index') or ('_noval' in c) or ('_onehot' in c):          
      continue 
    lgb_grid['use_'+c] = [True, False]
    if 'use_'+c not in best_params:
      best_params['use_'+c] = True



  best_loss = {'single' : 1e10, 'blended' : 1e10}
  blended_predictions = {'test' : [], 'validation' : []}
  for i in range(0,n_attempts):
      print('attempt ',i,'freq = ', freq)
      params = best_params.copy()
      if i > 0:
        params = update_params(params, lgb_grid)
      print('trying params :',params, 'freq = ', freq)
      gc.collect()
      loss, submission_format = process(freq, params, best_loss, train, submission_format, blended_predictions, submission_frequency, random_seed)  
      gc.collect()
      if loss['single'] < best_loss['single']:
        print('new best single loss', loss['single'],' found for freq = ',freq,'for best_lgb_params[',freq,'] = ', params)
        best_params = params.copy()
        best_loss['single'] = loss['single']
        if GLOBAL_PARAMS_UPDATE:
          best_lgb_params[freq] = params.copy()
      if loss['blended'] < best_loss['blended']:
        print('new best blended loss found, saving submission')
        best_loss['blended'] = loss['blended']
        submission_format[['obs_id','SiteId','Timestamp','ForecastId','Value']].to_csv(ODIR + 'lw'+str(OUTPUT_VERSION)+'_submission_lgb_'+str(random_seed)+'.csv', index = False)        
  return best_loss



#  Reading csv and conveting int64 to int32, float64 to float32 to decrease memory usage
#  Also some columns are renamed for compatibility between current preprocessing version and get_ratio_features method


def read32(filename):
  df_1 = pd.read_csv(filename, nrows = 1)
  dtp = dict(df_1.dtypes)
  for k in dtp:
    if dtp[k] == np.float64:
      dtp[k] = np.float32
    if dtp[k] == np.int64:
      dtp[k] = np.int32
  df = pd.read_csv(filename, dtype = dtp)
  if 'MeanBySiteIdDowTime10Y' in df.columns:
    new_to_old = {'MeanBySiteIdDowTime10Y' : 'MeanBySiteId',
                'MedianBySiteIdDowTime10Y' : 'MedianBySiteId',
                 'MaxBySiteIdDowTime10Y' : 'MaxBySiteId',
                 'MinBySiteIdDowTime10Y' : 'MinBySiteId',
                 'MeanByForecastIdDowTime10Y' : 'MeanByForecastId',
                 'MedianByForecastIdDowTime10Y' : 'MedianByForecastId',
                 'MaxByForecastIdDowTime10Y' : 'MaxByForecastId',
                 'MinByForecastIdDowTime10Y' : 'MinByForecastId',
#                 'MeanBySiteIdDowTime1W' : 'LastWeekValues',
#                 'MeanBySiteIdDowTime1W_offset1W' : 'PrevWeekValues',
                 'MeanByForecastIdDowTime1W' : 'LastWeekValues',
                 'MeanByForecastIdDowTime1W_offset1W' : 'PrevWeekValues',
                 'MeanBySiteIdTime1D' : 'LastDayValues'}
    for k in new_to_old.copy():
      new_to_old[k+'_noval'] = new_to_old[k]+'_noval'
    print('renaming :' , new_to_old)
    df.rename(columns=new_to_old, inplace = True)

  return df



# Averaging multiple csv files (predictions with different random seeds) into final prediction

def average(names):
  dfs = []
  for n in names:
    dfs.append( pd.read_csv(n))
  vs = [df['Value'].values for df in dfs]
  v = np.exp(np.mean(np.log(vs), axis = 0))
  dfs[0]['Value'] = v
  dfs[0].to_csv(ODIR + 'final_prediction.csv', index = False)



# Repeating training and prediction for each of 3 frequencies with different random seeds


def main():
  prtime('starting lgb.py PREPROCESS_VERSION =', PREPROCESS_VERSION, 'OUTPUT_VERSION = ', OUTPUT_VERSION)
  IDIR, ODIR = get_paths()
  train = read32(ODIR + 'train_updated_v'+str(PREPROCESS_VERSION)+'.csv')
  prtime('train reading done')
  gc.collect()
#  if LOG:
#    train.Value = np.log1p(train.Value)
  prtime('reading submission_format')
  submission_format = read32(ODIR + 'submission_format_updated_v'+str(PREPROCESS_VERSION)+'.csv')
  metadata = pd.read_csv(IDIR + 'metadata.csv').set_index('SiteId')

  prtime('generating features for submission')

  train = get_static_features(train, metadata)
  train = get_ratio_features(train)

  submission_format = get_static_features(submission_format, metadata)
  submission_format = get_ratio_features(submission_format)

  print(train.dtypes)
  print(train.memory_usage())
  print(submission_format.dtypes)
  print(submission_format.memory_usage())


  submission_frequency = pd.read_csv(IDIR + 'submission_frequency.csv')
#  submission_updated = pd.read_csv(ODIR + 'submission_updated.csv')
  train.Temperature.fillna(np.nanmedian(train.Temperature), inplace = True)
  submission_format.Temperature.fillna(np.nanmedian(train.Temperature), inplace = True)

  freqs = [900000000000, 3600000000000, 86400000000000]
  seeds = [14,15,16,17,18]


#  seeds = [14,15,16,17,18]*100
#  freqs = [86400000000000]

  best_losses = pd.DataFrame(columns = ['freq', 'seed', 'single', 'blended'], dtype = np.float32)

  for seed in seeds:
    if os.path.isfile(ODIR + 'lw'+str(OUTPUT_VERSION)+'_submission_lgb_'+str(seed)+'.csv'):
      print('skipping training for seed', seed,' file already exists')
      continue
    for freq in freqs:
      best_loss = tune_params(train, submission_format, submission_frequency, freq, n_attempts = NUM_ATTEMPTS, random_seed = seed)
      best_losses = best_losses.append({'freq' : freq, 'seed' : seed, 'single' : best_loss['single'], 'blended' : best_loss['blended']}, ignore_index = True)
      print('best losses so far = ', best_losses)
      print(best_losses.groupby('freq')['single','blended'].mean())
#      print('last 5 losses mean : ', best_losses.iloc[-5:]['blended'].mean())
  filenames = [ODIR + 'lw'+str(OUTPUT_VERSION)+'_submission_lgb_'+str(seed)+'.csv' for seed in seeds]
  average(filenames)   
  
if __name__ == '__main__':
  main()

      





