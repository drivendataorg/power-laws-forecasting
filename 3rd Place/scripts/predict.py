print('Starting')

# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Imputing missing values in temp and value
from sklearn.preprocessing import Imputer

# Best practice to scale features
from sklearn.preprocessing import MinMaxScaler

# Models used for prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from itertools import chain

# Turn off setting with copy warning
pd.options.mode.chained_assignment = None

print('Loading Data')

# Read in the dataframes for training and testing
train = pd.read_csv('../data/train_corrected.csv')
test = pd.read_csv('../data/test_corrected.csv')

# Convert to datetimes
train['Timestamp'] = pd.to_datetime(train['Timestamp'])
test['Timestamp'] = pd.to_datetime(test['Timestamp'])

print('Data Loaded')

# Takes in a site id and returns a formatted training and testing set
def process(site):

	# Testing data
	test_df = test[test['ForecastId'] == site].sort_values(['Timestamp', 'Distance'])
	test_df = test_df.drop_duplicates(['Timestamp'], keep='first')

	# Training data
	train_df = train[train['ForecastId'] == site].sort_values(['Timestamp', 'Distance'])
	train_df = train_df.drop_duplicates(['Timestamp'], keep='first')

	# Only use past training data
	train_df = train_df[train_df['Timestamp'] < test_df['Timestamp'].min()]

	# If all training temperatures are missing, drop temperatures from both training and testing
	if (np.all(np.isnan(train_df['Temperature']))) or (np.all(np.isnan(test_df['Temperature']))):
		train_df = train_df.drop(labels = 'Temperature', axis=1)
		test_df = test_df.drop(labels= 'Temperature', axis=1)

	# Otherwise impute the missing temperatures
	else:
		temp_median_imputer = Imputer(missing_values='NaN', strategy='median', axis = 0)
		temp_median_imputer.fit(train_df[['Temperature']])
		train_df['Temperature'] = temp_median_imputer.transform(train_df[['Temperature']])
		test_df['Temperature'] = temp_median_imputer.transform(test_df[['Temperature']])

	# Impute the missing values
	value_median_imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
	value_median_imputer.fit(train_df[['Value']])

	if pd.isnull(train_df['Value']).all():
		train_df['Value'] = 0
	else:
		train_df['Value'] = value_median_imputer.transform(train_df[['Value']])

	# Find the minimum date for converting timestamp to numeric
	min_date = min(train_df['Timestamp'])

	# Convert timestamp to numeric
	train_df['Timestamp'] = (train_df['Timestamp'] - min_date).dt.total_seconds()
	test_df['Timestamp']  = (test_df['Timestamp'] - min_date).dt.total_seconds()

	# Interval between measurements
	train_df['time_diff'] = train_df['Timestamp'].diff().fillna(0)
	test_df['time_diff'] = test_df['Timestamp'].diff().fillna(0)

	# Extract labels
	train_labels = train_df['Value']

	# Drop columns
	train_df = train_df.drop(columns = ['Distance', 'SiteId', 'ForecastId', 'Value'])
	test_df =   test_df.drop(columns = ['Distance', 'SiteId', 'ForecastId', 'Value'])


	# Scale the features between 0 and 1 (best practice for ML)
	scaler = MinMaxScaler()

	train_df.ix[:, :] = scaler.fit_transform(train_df.ix[:, :])
	test_df.ix[:, :] = scaler.transform(test_df.ix[:, :])

	return train_df, train_labels, test_df


# Trains and predicts for all datasets, makes predictions one site at a time
def predict():

	# List of trees to use in the random forest and extra trees model
	trees_list = list(range(50, 176, 25))

	# List of site ids
	site_list = list(set(train['ForecastId']))

	predictions = []

	# Keep track of the sites run so far
	number = len(site_list)
	count = 0

	# Iterate through every site
	for site in site_list:

		# Features and labels
		train_x, train_y, test_x = process(site)

		# Make sure only training on past data
		assert train_x['Timestamp'].max() < test_x['Timestamp'].min(), 'Training Data Must Come Before Testing Data'

		# Initialize list of predictions for site
		_predictions = np.array([0. for _ in range(len(test_x))])

		# Iterate through the number of trees
		for tree in trees_list:

			# Create a random forest and extra trees model with the number of trees
			model1 = RandomForestRegressor(n_estimators=tree, n_jobs=-1)
			model2 = ExtraTreesRegressor(n_estimators=tree, n_jobs=-1)

			# Fitting the model
			model1.fit(train_x, train_y)
			model2.fit(train_x, train_y)

			# Make predictions with each model
			_predictions += np.array(model1.predict(test_x))
			_predictions += np.array(model2.predict(test_x))

		# Average the predictions
		_predictions = _predictions / (len(trees_list) * 2)

		# Add the predictions to the list of all predictions
		predictions.append(list(_predictions))

		# Iterate the count
		count = count + 1

		# Keep track of number of buildings process so far
		if count % 100 == 0:
			print('Percentage Complete: {:.1f}%.'.format(100 * count / number))

	# Flatten the list
	predictions = list(chain(*predictions))

	return predictions

# Make a submission file given the list of predictions and name for the submission
def make_submission_file(predictions, name):

	# Read in the submission dataframe
	submit_df = pd.read_csv('../data/submission_format.csv')

	# Assign the predictions as the value
	submit_df['Value'] = predictions

	# Save the submissions to the folder of final submissions
	submit_df.to_csv('../submissions/%s.csv' % name, index = False)
	print('Predictions saved to ../submissions/%s.csv' % name)

print('Starting Predictions')

# Make predictions
predictions = predict()

# Save predictions with a sensible name
make_submission_file(predictions, 'submission_filename')

