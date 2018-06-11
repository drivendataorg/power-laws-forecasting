# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Original data
train = pd.read_csv('../data/train.csv', index_col=0)
test = pd.read_csv('../data/submission_format.csv', index_col=0)

weather = pd.read_csv('../data/weather.csv', index_col=0)
meta = pd.read_csv('../data/metadata.csv')

# Extrac features and convert the time into cyclical variables
def process_time(df):
    
    # Convert timestamp into a pandas datatime object
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    
    # Extract units of time from the timestamp
    df['min'] = df.index.minute
    df['hour'] = df.index.hour
    df['wday'] = df.index.dayofweek
    df['mday'] = df.index.day
    df['yday'] = df.index.dayofyear
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # Create a time of day to represent hours and minutes
    df['time'] = df['hour'] + (df['min'] / 60)
    df = df.drop(columns=['hour', 'min'])
    
    # Cyclical variable transformations
    
    # wday has period of 6
    df['wday_sin'] = np.sin(2 * np.pi * df['wday'] / 6)
    df['wday_cos'] = np.cos(2 * np.pi * df['wday'] / 6)
    
    # yday has period of 365
    df['yday_sin'] = np.sin(2 * np.pi * df['yday'] / 365)
    df['yday_cos'] = np.cos(2 * np.pi * df['yday'] / 365)
    
    # month has period of 12
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # time has period of 24
    df['time_sin'] = np.sin(2 * np.pi * df['time'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['time'] / 24)
    
    # turn the index into a column
    df = df.reset_index(level=0)
    
    return df
  
# Feature engineering of the time for train and test
train = process_time(train)
test = process_time(test)

# Function to add weather information into a dataset
def add_weather(df, weather):
    
    # Keep track of the original length of the dataset
    original_length = len(df)
    
    # Convert timestamp to a pandas datetime object
    weather['Timestamp'] = pd.to_datetime(weather['Timestamp'])
    weather = weather.set_index('Timestamp')
    
    # Round the  weather data to the nearest 15 minutes
    weather.index = weather.index.round(freq='15 min')
    weather = weather.reset_index(level=0)
    
    # Merge the building data with the weather data
    df = pd.merge(df, weather, how = 'left', on = ['Timestamp', 'SiteId'])
    
    # Drop the duplicate temperature measurements, keeping the closest location
    df = df.sort_values(['Timestamp', 'SiteId', 'Distance'])
    df = df.drop_duplicates(['Timestamp', 'SiteId'], keep='first')
    
    # Checking length of new data
    new_length = len(df)
    
    # Check to make sure the length of the dataset has not changed
    assert original_length == new_length, 'New Length must match original length'

    return df

# Get weather information for both train and test data
train = add_weather(train, weather)
test = add_weather(test, weather)

# List of ids and new dataframe to hold meta information
id_list = set(meta['SiteId'])
all_meta = pd.DataFrame(columns=['SiteId', 'wday', 'off'])

# Iterate through each site and find days off
for site in id_list:
    # Extract the metadata information for the site
    meta_slice = meta.ix[meta['SiteId'] == site]
    
    # Create a new dataframe for the site
    site_meta = pd.DataFrame(columns=['SiteId', 'wday', 'off'],
                            index = [0, 1, 2, 3, 4, 5, 6])
    
    site_meta['wday'] = [0, 1, 2, 3, 4, 5, 6]
    site_meta['SiteId'] = site
    
    # Record the days off
    site_meta.ix[0, 'off'] = float(meta_slice['MondayIsDayOff'])
    site_meta.ix[1, 'off'] = float(meta_slice['TuesdayIsDayOff'])
    site_meta.ix[2, 'off'] = float(meta_slice['WednesdayIsDayOff'])
    site_meta.ix[3, 'off'] = float(meta_slice['ThursdayIsDayOff'])
    site_meta.ix[4, 'off'] = float(meta_slice['FridayIsDayOff'])
    site_meta.ix[5, 'off'] = float(meta_slice['SaturdayIsDayOff'])
    site_meta.ix[6, 'off'] = float(meta_slice['SundayIsDayOff'])
    
    # Append the resulting dataframe to all site dataframe
    all_meta = all_meta.append(site_meta) 
    
# Find the days off in the training and testing data
train = train.merge(all_meta, how = 'left', on = ['SiteId', 'wday'])
test = test.merge(all_meta, how = 'left', on = ['SiteId', 'wday'])

# Save files to csv
train.to_csv('../data/train_corrected.csv', index = False)
test.to_csv('../data/test_corrected.csv', index = False)