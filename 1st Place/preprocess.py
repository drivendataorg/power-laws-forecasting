import pandas as pd
import numpy as np
from util import get_paths, prtime
import multiprocessing
from multiprocessing import Process, Queue
import time
import traceback
import sys
import gc
import lgb
import os
from collections import OrderedDict

IDIR, ODIR = get_paths()
VERSION = 12                              # Version of preprocessed files, used in output files names like 'train_updated_vNN.csv'


# Number of nan values in an array

def my_nancount(a):
  return np.sum(np.isnan(a))



#  Calculating historical aggregates
#  df - source dataframe (train set)
#  TestTimestamp - start of test period, no data at this point or beyond is used
#  period - amount of time before TestTimestamp used to calculate aggregetes
#  target col - column to calculate averages (can be Value, Temperature, ...)
#  cols - columns to group by (i.e. we are getting aggregate values for the same values in these columns in the past
#  col_values - current values in this columns (for example, current time and day of week)

#  Notice : in its current state this function relies on Timestamp values being sorted (ascending) within each group,
#  so can't be used for aggregates over different SiteIds


def get_aggregates(df, TestTimestamp, period, target_col, cols, func_list, offset_name, col_values, group_cache, noval_name = ''):
  
#  prtime('cols = ', cols)
  start = time.time()

#  print('gagc = ', get_aggregates.gb_cache)
  if (tuple(cols), target_col, col_values) in group_cache:
    subset = group_cache[(tuple(cols), target_col, col_values)]
  else:
    if tuple(cols) in get_aggregates.gb_cache:
      gb = get_aggregates.gb_cache[tuple(cols)]
    else:
      if len(cols):
        gb = df.groupby(cols)['Value','Temperature']
        get_aggregates.gb_cache[tuple(cols)] = gb
    
    if len(cols):
      if col_values in gb.groups.keys():
        subset = gb.get_group(col_values)[target_col] #.set_index('Timestamp')  # Slice with the current values in the corresponding columns
      else:
        subset = df.iloc[0:0] # empty slice
    else:
      subset = df # No slicing, using all data
    group_cache[(tuple(cols), target_col, col_values)] = subset
  get_aggregates.times['get_group'] += time.time()-start
  start = time.time()

  first_idx = np.searchsorted(subset.index, TestTimestamp - period[0])
  last_idx = np.searchsorted(subset.index, TestTimestamp)
  subset = subset.values[first_idx : last_idx]
  get_aggregates.times['subset'] += time.time()-start
  start = time.time()


  postfix = (''.join(cols))+period[1]

  res = OrderedDict()
  for func in func_list:
    if target_col != 'Value':
      agg_name = func[1]+target_col+'By'+postfix+offset_name+noval_name
    else:
      agg_name = func[1]+'By'+postfix+offset_name+noval_name
    if len(subset) == 0: 
      res[agg_name] = np.nan   # zero-len subset, unable to calculate
    else:
      res[agg_name] = func[0](subset)  # calculating

  get_aggregates.times['aggs'] += time.time()-start
  return res
get_aggregates.gb_cache = {}   # static variable with groupby caches
get_aggregates.times = {'get_group' : 0.0, 'subset' :0.0, 'aggs' : 0.0}




# Calculating aggegates for a given forecast id for both train and sample_submission (test) datasets
# For the train set, two types of aggregates are calculated : random ones and _noval.
# Random aggregates use period of time with some random lag from current sample's timestamp, this lag
# is used to simulate test situation when past history is available only with some lag for each sample
# Noval aggregates are calculated for last part of each ForecastId's data, and are used for validation :
# they are calculated with lag increasing from 0 to forecast lenght, so no consumption data after validation
# period start is used


def process_forecastid(i, train, submission, filters):

    submission_subset = submission.loc[submission.ForecastId == i]
    train_subset = train.loc[train.ForecastId == i]

    SAVE_MEMORY = True
    if SAVE_MEMORY:
      SiteId = train_subset['SiteId'].iat[0]
      train_to_agg = train.loc[train.SiteId == SiteId]
    else:
      train_to_agg = train

    submission_res = []
    train_res = []
    group_cache = {}
    forecast_len = len(submission_subset)
    time_step = submission_subset['Timestamp'].iat[1]-submission_subset['Timestamp'].iat[0] 

    dist_from_start = 0

#   Processing train

    noval_Timestamp = train_subset.loc[(train_subset.Validation == 1)].Timestamp.min()  # First sample of validation period
    lag_noval = 0
    for idx, row in train_subset.iterrows(): 
      aggs = OrderedDict()
      lag = np.random.choice(forecast_len)   # random lag for each train sample
      aggs['lag'] = lag
      aggs['dist_from_start'] = dist_from_start
      dist_from_start += 1
      lag_time = lag*time_step
      for c, p, target_col, offs, func_list in filters:
        if len(c):
          vals = row[list(c)]  
          vals = tuple(vals) if len(vals) > 1 else tuple(vals)[0]
        else:
          vals = None
        col_aggs = get_aggregates(train_to_agg, row.Timestamp-offs[0]-lag_time, p, target_col, c, func_list, offs[1], vals, group_cache)        
        aggs.update(col_aggs)
        if row.Validation == 1:           # if we need to calculate noval aggregates as well
          noval_aggs = get_aggregates(train_to_agg, noval_Timestamp-offs[0], p, target_col, c, func_list, offs[1], vals, group_cache, '_noval')        
          noval_aggs['lag_noval'] = lag_noval
          lag_noval += 1
          aggs.update(noval_aggs)

      train_res.append((row.original_index, aggs))


 #  Processing submission

    lag = 0

    TestTimestamp = submission_subset['Timestamp'].min()  # start of forecast period
    for idx, row in submission_subset.iterrows(): 
      aggs = OrderedDict()
      aggs['lag'] = lag
      aggs['dist_from_start'] = dist_from_start
      lag += 1
      dist_from_start += 1
      for c, p, target_col, offs, func_list in filters:
        if len(c):
          vals = row[list(c)]  
          vals = tuple(vals) if len(vals) > 1 else tuple(vals)[0]
        else:
          vals = None
        col_aggs = get_aggregates(train_to_agg, TestTimestamp-offs[0], p, target_col, c, func_list, offs[1],vals, group_cache)        
        aggs.update(col_aggs)
      submission_res.append((idx, aggs))


#      prtime('idx = ', idx)
    if SAVE_MEMORY:
      get_aggregates.gb_cache.clear()             # Clearing cache before processing next ForecastId
      
    print('times = ', get_aggregates.times)
    return train_res, submission_res



# Worker, used for multiprocessing. In each process this function gets new tasks (forecast_ids) from input queue and puts
# processed data (calculated aggregates) to output queue.


def forecastid_worker(train, submission, filters, q_in, q_out, worker_id):
  while(not q_in.empty()):
    try:
      forecast_id = q_in.get(timeout=60)
    except BaseException as e:
      print('Exception : ', str(e))
      print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
      print(traceback.format_exc())
      sys.stdout.flush()
      break
    prtime('worker', worker_id, 'processing forecast_id:', forecast_id)
#    aggs = None    
    train_res, submission_res = process_forecastid(forecast_id, train, submission, filters)
    gc.collect()
    q_out.put((forecast_id, train_res, submission_res))
  q_out.put((-1, None, None))
  prtime('worker', worker_id, 'processing done, exiting')



#  Calculating validation status. Last submission_len samples for each forecast_id get status of 1


def get_validation(train, submission):
    forecast_ids = train.ForecastId.unique()

    # optimized version
    validation = np.zeros(len(train))
    train_lens = train.groupby('ForecastId').size()
    submission_lens = submission.groupby('ForecastId').size()
    prtime('validation groupbys done')      
    cur_idx = 0
    for i in forecast_ids:
      train_len = train_lens[i]
      submission_len = submission_lens[i]
      submission_len = 192 if submission_len > 100 else 60   # to make validation sizes the same for each frequency
      validation[cur_idx+train_len-submission_len:cur_idx+train_len] = 1
      cur_idx += train_len
    return validation.astype(np.int8)




# Main function used to calculate aggregate values and store them as columns in train and submission_format dataframes


def get_means_new(train, submission):
  train['Timestamp'] = pd.to_datetime(train['Timestamp'])
  train['Value'] = train['Value'].astype(np.float32)
  train['Temperature'] = train['Temperature'].astype(np.float32)
  train['SiteId'] = train['SiteId'].astype(np.int16)
  train['ForecastId'] = train['ForecastId'].astype(np.int16)
  train['IsDayOff'] = train['IsDayOff'].astype(np.int8)
  train['IsDayOff2'] = train['IsDayOff2'].astype(np.int8)
  train['IsDayOff3'] = train['IsDayOff3'].astype(np.int8)

  submission['Timestamp'] = pd.to_datetime(submission['Timestamp'])
  train['Dow'] = train['Timestamp'].dt.dayofweek.astype(np.int8)
  submission['Dow'] = submission['Timestamp'].dt.dayofweek.astype(np.int8)
  train['Time'] = train.Timestamp.dt.time  # extracting time without date
  submission['Time'] = submission.Timestamp.dt.time
  train['Validation'] = get_validation(train, submission)


  train['original_index'] = train.index
  train.set_index('Timestamp', inplace = True, drop = False)
  column_filter = ['SiteId','ForecastId']
  columns_gb = ['',['Time'],['Dow'],['Time','Dow']]
  cols = []
  for cf in column_filter:
    for cgb in columns_gb:
      c = []
      if len(cf): 
        c += [cf] 
      if len(cgb): 
        c += cgb
      cols.append(tuple(c) if len(c) > 1 else c)
  print('cols = ', cols)
  periods = [(np.timedelta64(1,'D'),'1D'), (np.timedelta64(3,'D'),'3D'), 
    (np.timedelta64(1,'W'),'1W'),  (np.timedelta64(1,'M'),'1M'), 
    (np.timedelta64(1,'Y'),'1Y'), (np.timedelta64(10,'Y'),'10Y')]


  func_list = [(np.nanmean,'Mean'), (np.nanmedian, 'Median'), (np.nanmax, 'Max'), (np.nanmin, 'Min')]



  filters = [] 
  zero = (np.timedelta64(0,'D'),'')
  oneweek = (np.timedelta64(1,'W'),'_offset1W')
  mean_only = [func_list[0]]
  median_only = [func_list[1]]
  nan_count = [(my_nancount,'nancount')]
  filters.append((['SiteId','Dow','Time'], (np.timedelta64(10,'Y'),'10Y'), 'Value', zero, func_list))
  filters.append((['ForecastId','Dow','Time'], (np.timedelta64(10,'Y'),'10Y'), 'Value', zero, func_list))

  filters.append((['SiteId','IsDayOff','Time'], (np.timedelta64(10,'Y'),'10Y'), 'Value', zero, func_list))
  filters.append((['ForecastId','IsDayOff','Time'], (np.timedelta64(10,'Y'),'10Y'), 'Value', zero, func_list))

  filters.append((['SiteId','IsDayOff2','Time'], (np.timedelta64(10,'Y'),'10Y'), 'Value', zero, func_list))
  filters.append((['SiteId','IsDayOff2','Time'], (np.timedelta64(10,'Y'),'10Y'), 'Value', oneweek, median_only))

  filters.append((['SiteId','IsDayOff3','Time'], (np.timedelta64(10,'Y'),'10Y'), 'Value', zero, median_only))
  filters.append((['SiteId','IsDayOff2','Dow','Time'], (np.timedelta64(10,'Y'),'10Y'), 'Value', zero, median_only))
  filters.append((['SiteId','IsDayOff3','Dow','Time'], (np.timedelta64(10,'Y'),'10Y'), 'Value', zero, median_only))


  filters.append((['SiteId'], (np.timedelta64(10,'Y'),'10Y'), 'Value', zero, func_list))
  filters.append((['SiteId'], (np.timedelta64(1,'Y'),'1Y'), 'Value', zero, func_list))
  filters.append((['SiteId'], (np.timedelta64(1,'M'),'1M'), 'Value', zero, func_list))
  filters.append((['SiteId'], (np.timedelta64(1,'W'),'1W'), 'Value', zero, func_list))

  filters.append((['SiteId','Dow'], (np.timedelta64(10,'Y'),'10Y'), 'Value', zero, func_list))
  filters.append((['ForecastId','Dow'], (np.timedelta64(10,'Y'),'10Y'),'Value', zero, func_list))
  filters.append((['SiteId','Time'], (np.timedelta64(10,'Y'),'10Y'),'Value', zero, func_list))
  filters.append((['ForecastId','Time'], (np.timedelta64(10,'Y'),'10Y'),'Value', zero, func_list))


#  filters.append((['SiteId','Dow','Time'], (np.timedelta64(1,'W'),'1W'),zero, mean_only))  # LastWeekValues
#  filters.append((['SiteId','Dow','Time'], (np.timedelta64(1,'W'),'1W'),(np.timedelta64(1,'W'),'_offset1W'), mean_only))  # PrevWeekValues

  filters.append((['ForecastId','Dow','Time'], (np.timedelta64(1,'W'),'1W'),'Value', zero, mean_only))  # LastWeekValues
  filters.append((['ForecastId','Dow','Time'], (np.timedelta64(1,'W'),'1W'),'Value', oneweek, mean_only))  # PrevWeekValues
  filters.append((['SiteId','Time'], (np.timedelta64(1,'D'),'1D'),'Value', zero, mean_only))  # LastDayValues
  filters.append((['SiteId','Time'], (np.timedelta64(1,'D'),'1D'),'Value', oneweek, mean_only))  # Last day values with one week offset

  filters.append((['SiteId'], (np.timedelta64(1,'h'),'1h'),'Value', zero, mean_only))  # LastHourValues
  filters.append((['SiteId'], (np.timedelta64(1,'h'),'1h'),'Value', oneweek, mean_only))  # LastHourValues with one week offset

  filters.append((['SiteId'], (np.timedelta64(1,'D'),'1D'),'Value', zero, nan_count))  # Nans in last day
  filters.append((['SiteId'], (np.timedelta64(1,'W'),'1W'),'Value', zero, nan_count))  # Nans in week
  filters.append((['SiteId'], (np.timedelta64(1,'M'),'1M'),'Value', zero, nan_count))  # Nans in month
  filters.append((['SiteId'], (np.timedelta64(10,'Y'),'10Y'),'Value', zero, nan_count))  # Nans in 10 years
  
  filters.append((['SiteId','Time'], (np.timedelta64(10,'Y'),'10Y'), 'Temperature', zero, median_only)) # Median temperature at this time
  filters.append((['SiteId','Time'], (np.timedelta64(1,'M'),'1M'), 'Temperature', zero, median_only)) # Median temperature at this time last month




  # creating submission aggregates

  ids = submission.ForecastId.unique()
  PARALLEL = True
  if PARALLEL:
    q_in = Queue()
    for i in ids:
      q_in.put(i)
    q_out = Queue()
    all_processes = []
    cpu_count = min(15,max(1,multiprocessing.cpu_count()))
    for i in range(cpu_count):
      all_processes.append(Process(target=forecastid_worker, args=(train, submission, filters, q_in, q_out, i)))
    for p in all_processes:
      p.start()
    finished_count = 0
    while True:
      forecastid, train_res, submission_res = q_out.get()
      if forecastid == -1:
        finished_count +=1
        prtime('finished workers :' , finished_count)
        if finished_count == cpu_count:
          break
      else:
        prtime('updating submission with calculated aggregates for forecastid = ', forecastid)
        for idx, aggs in submission_res:
          for a in aggs:
            if a not in submission.columns:
              prtime('adding new column to submission :',a)
              submission[a] = np.float32(np.nan)   # adding new column if necessary
              prtime('column adding done')
#            prtime('setting value', aggs[a], type(aggs[a]), ' at ', int(idx), a)
            submission[a].iat[int(idx)] = aggs[a]
#            prtime('setting value done')

        prtime('updating train with calculated aggregates for forecastid = ', forecastid)
        for idx, aggs in train_res:
          for a in aggs:
            if a not in train.columns:
              prtime('adding new column to train :',a)
              train[a] = np.float32(np.nan)   # adding new column if necessary
            train[a].iat[int(idx)] = aggs[a]


        prtime('updating train done')
        
    for p in all_processes:
      p.join()
      prtime('p.join done')
  else:
    raise NotImplementedError

  return train, submission





# Preprocessing : for each sample in train and test sets, calculates features like IsHoliday, IsDayOff, Temperature (at the nearest station),
# etc. When the first part is done (it is relatively fast), calls get_means_new to calculate historical aggregates


def main(update_means_only = False, forced_update = True):
  if forced_update == False:
    if os.path.isfile(ODIR + 'train_updated_v'+str(VERSION)+'.csv') and os.path.isfile(ODIR + 'submission_format_updated_v'+str(VERSION)+'.csv'):
      print('preprocessed train and submission_format files already exist, forced_update == False: skipping preprocessing')
      return
  prtime('starting preprocessing')

  # Reading original data files
 

  if update_means_only:
    train = pd.read_csv(ODIR + 'train_updated_temp_2.csv')
    submission_format = pd.read_csv(ODIR + 'submission_format_updated_temp_2.csv')
  else:
    prtime('reading data')
    train = pd.read_csv(IDIR + 'train.csv')
    submission_format = pd.read_csv(IDIR + 'submission_format.csv')
    metadata = pd.read_csv(IDIR + 'metadata.csv').set_index('SiteId')
    weather = pd.read_csv(IDIR + 'weather.csv')# .set_index('SiteId')
    holidays = pd.read_csv(IDIR + 'holidays.csv')
    prtime('reading done')

    holidays['Date'] = pd.to_datetime(holidays['Date'])
    holidays_by_date = holidays.groupby('Date')['Holiday'].count()
    holidays.set_index(['SiteId','Date'], inplace = True)
    holidays_unique = holidays[~holidays.index.duplicated(keep='first')]
    

    weather['Timestamp'] = pd.to_datetime(weather['Timestamp'])
    median_temp = weather.groupby('Timestamp')['Temperature'].median().reset_index()                  # Median across all sites
      

    dfs = [(train,'train'),(submission_format,'submission_format')]

    for df, name in dfs:
      df['Timestamp'] = pd.to_datetime(df['Timestamp'])
      df['Date'] = pd.to_datetime(df['Timestamp'].dt.date)
      df['Dow'] = df['Timestamp'].dt.dayofweek
      df['Temperature'] = np.float32(np.nan)

      prtime('calculating weather for', name)
      for i in df.SiteId.unique():
        prtime('processing siteid ', i)
        weather_subset = weather.loc[weather.SiteId == i]
        if len(weather_subset) == 0:
          weather_subset = median_temp
#        print('weather_subset =', weather_subset)
        merged = pd.merge_asof(df.loc[df.SiteId == i], weather_subset[['Timestamp','Temperature']], on = 'Timestamp', direction = 'nearest')           
#        print('merged = ', merged)
        df.loc[df.SiteId == i, 'Temperature'] = merged.Temperature_y.values
       
      
#      df['Temperature'] = df.apply(GetNearestWeather, axis = 1)
      df.to_csv(ODIR + name+'_updated_temp_1.csv', index = False)


      prtime('calculating IsHoliday for', name)

      site_date = pd.Series(list(zip(df.SiteId, df.Date)), index = df.index)
      site_date_yest = pd.Series(list(zip(df.SiteId, df.Date-np.timedelta64(1,'D'))), index = df.index)
      site_date_tom = pd.Series(list(zip(df.SiteId, df.Date+np.timedelta64(1,'D'))), index = df.index)
      df['IsHoliday'] = site_date.map(holidays_unique['Holiday']).notnull()     
      df['IsHolidayYesterday'] = site_date_yest.map(holidays_unique['Holiday']).notnull()     
      df['IsHolidayTomorrow'] = site_date_tom.map(holidays_unique['Holiday']).notnull()     


      prtime('calculating HolidaysSomewhere for', name)
#      df['HolidaysSomewhere'] = df.apply(HolidaysSomewhere, axis = 1)
      df['HolidaysSomewhere'] = df['Date'].map(holidays_by_date).fillna(0)
      df['HolidaysSomewhereYesterday'] = (df['Date']-np.timedelta64(1,'D')).map(holidays_by_date).fillna(0)
      df['HolidaysSomewhereTomorrow'] = (df['Date']+np.timedelta64(1,'D')).map(holidays_by_date).fillna(0)


      prtime('Calculating IsDayOff for', name)
      dayoff_columns = ['MondayIsDayOff','TuesdayIsDayOff','WednesdayIsDayOff','ThursdayIsDayOff','FridayIsDayOff','SaturdayIsDayOff','SundayIsDayOff']      
      df['IsDayOff'] = False
      for siteid in df.SiteId.unique():
        prtime('processing siteid = ', siteid)
        isdayoff = metadata.loc[siteid]
        for idx, col_name in enumerate(dayoff_columns):
          if idx < 4:                                     # Always False in this metadata set, no need to process
            continue                                      
          df.loc[(df.SiteId == siteid) & (df.Dow == idx), 'IsDayOff'] = isdayoff[col_name]
      df['IsDayOff2'] = (df['IsDayOff'] > 0) | (df['IsHoliday'] > 0)
      df['IsDayOff3'] = (df['IsDayOff'] > 0) | (df['HolidaysSomewhere'] >0)
      df.to_csv(ODIR + name+'_updated_temp_2.csv', index = False)  


#      df['IsHoliday']  = df.apply(IsHoliday, axis = 1)




  prtime('calculating means')
  train,submission_format = get_means_new(train, submission_format)

  train.to_csv(ODIR + 'train_updated_v'+str(VERSION)+'.csv', index = False)
  submission_format.to_csv(ODIR + 'submission_format_updated_v'+str(VERSION)+'.csv', index = False)

  print('preprocessing done')


if __name__ == '__main__':
  main(update_means_only = False)
  lgb.main()
