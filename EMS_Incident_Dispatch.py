import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(pd.__version__)

# Fixing the issue by correcting the file path to the available file
# Datafile location
infile="911_End-to-End_Data (3).csv"

df = pd.read_csv(infile)

# Display the first few records to verify data loaded successfully
df.head()

#Checking number of rows # Calculate and print the number of rows in the dataset
num_rows = df.shape[0]
print(f"The dataset contains {num_rows} rows.")

df.head()

from IPython.display import display, HTML

table_html = '<table style="border-collapse: collapse;"><tr><th>Column Name</th><th>Data Type</th><th>Value Count</th><th>Null Count</th></tr>'
for col in df.columns:
    dtype = df[col].dtype
    value_count = df[col].value_counts().shape[0]
    null_count = df[col].isnull().sum()
    table_html += f'<tr><td style="border: 1px solid black; padding: 5px;">{col}</td><td style="border: 1px solid black; padding: 5px;">{dtype}</td><td style="border: 1px solid black; padding: 5px;">{value_count}</td><td style="border: 1px solid black; padding: 5px;">{null_count}</td></tr>'
table_html += '</table>'

display(HTML(table_html))

# Filter out rows where 'AgencyCallPickup' is NA
df = df[df['Incident Type'].notna()]

#Selecting only the key variables
EMS = df[['Responding Agency', 'Incident Type', 'Number of Incidents', 
                'Average Call to Pickup Time', 'Dispatch Time', 'Response Time']]

EMS.head()

#Converting time columns to minutes from seconds
time_columns = ['Average Call to Pickup Time', 'Dispatch Time', 'Response Time']

# Convert the columns from seconds to minutes and round to 2 decimal places using .loc
for col in time_columns:
    EMS.loc[:, col] = (EMS[col] / 60).round(2)  # Use .loc to explicitly set the values



EMS.head()

import pandas as pd
# Count NA in EMS Data
total_na = EMS.isna().sum().sum()
print(f"Total missing values: {total_na}")

category_mapping = {
    '2. Non-Structural Fires': 'Fires',
    '1. Critical': 'Medical Emergencies',
    'Dispute': 'Dispute',
    'Other Crimes (In Progress)': 'Crime',
    '3. Non-Critical': 'Medical Emergencies',
    '3. Medical Emergencies': 'Medical Emergencies',
    '2. Serious': 'Medical Emergencies',
    '1. Life Threating Med Emergencies': 'Medical Emergencies',
    'Possible Crimes': 'Crime',
    'Past Crime': 'Crime',
    'Alarms': 'Non-Medical Emergencies',
    '1. Structural Fires': 'Fires',
    '2. Non-Life Threatening Med Emergencies': 'Medical Emergencies',
    '4. Non-Medical Emergencies': 'Non-Medical Emergencies',
    'Police Officer/Security Holding Suspect': 'Crime',
    'Investigate/Possible Crime': 'Crime',
    'Disorderly Person/Group/Noise': 'Dispute',
    'Vehicle Accident': 'Vehicle Accident',
    'Hazardous Materials/Suspicious Letters/Packages/Substances/Substances': 'Non-Medical Emergencies',
    'Shot Spotter': 'Crime',
    'United States Postal Service - Bio Hazard Detection System': 'Non-Medical Emergencies'
}

# Default category for unmapped values
default_category = 'Unknown'

# Use .loc[] for modifying the DataFrame
EMS.loc[:, 'Incident Type'] = EMS['Incident Type'].map(category_mapping).fillna(default_category)

# Create dummy variables
incident_dummies = pd.get_dummies(EMS['Incident Type'], prefix='Category')

# Optionally, merge the dummies with the original DataFrame
EMS = pd.concat([EMS, incident_dummies], axis=1)

#Displaying the Final Incident Type Categories
EMS['Incident Type'].unique()

# Make all numeric values absolute (Removing 8 negative values were identified across the dataset)
numeric_cols = EMS.select_dtypes(include=['number']).abs()

# Generate descriptive statistics for numeric columns
descriptive_stats = numeric_cols.describe(percentiles=[0.5]).T

# Add the count (n) and median explicitly
descriptive_stats['n'] = numeric_cols.count()
descriptive_stats['median'] = numeric_cols.median()

# Select and reorder the desired statistics
descriptive_stats = descriptive_stats[['n', 'mean', 'std', 'min', 'median', 'max']]

# Round the values to 2 decimal places
descriptive_stats = descriptive_stats.round(2)

# Print the table with a title
print("Table 1. Descriptive Statistics")
print(descriptive_stats)


import matplotlib.pyplot as plt
import seaborn as sns

# Histogram for Response Time
plt.figure(figsize=(8, 6))
sns.histplot(EMS['Response Time'], bins=30, kde=True, color='blue')
plt.title('Fig.1 Distribution of Response Time(in Minutes)')
plt.xlabel('Response Time (Minutes)')
plt.ylabel('Frequency')
plt.show()

# Bar chart for Number of Incidents by Incident Type (horizontal)
incident_counts = EMS.groupby('Incident Type')['Number of Incidents'].sum().sort_values()

plt.figure(figsize=(7, 5))
incident_counts.plot(kind='barh', color='orange')
plt.title('Fig.2 Number of Incidents by Incident Type')
plt.xlabel('Number of Incidents')
plt.ylabel('Incident Type')
plt.show()

# Boxplot for Response Time by Agent
plt.figure(figsize=(7, 5))
sns.boxplot(data=EMS, x='Responding Agency', y='Response Time', palette='Set2')
plt.title('Fig 3 Response Time by Agency')
plt.xlabel('Agency')
plt.ylabel('Response Time (in Minutes)')
plt.xticks(rotation=45)
plt.show()

# Boxplot for Response Time by Agent
plt.figure(figsize=(7, 5))
sns.boxplot(data=EMS, x='Incident Type', y='Response Time', palette='Set2')
plt.title('Fig 4 Response Time by Incident Type')
plt.xlabel('Incident Type')
plt.ylabel('Response Time (in Minutes)')
plt.xticks(rotation=45)
plt.show()

# Scatter plot: Response Time vs. Number of Incidents
plt.figure(figsize=(8, 6))
sns.scatterplot(data=EMS, x='Number of Incidents', y='Response Time', hue='Responding Agency', palette='viridis')
plt.title('Fig. 5 Response (Minutes) vs. Number of Incidents')
plt.xlabel('Number of Incidents')
plt.ylabel('Response Time (Minutes)')
plt.legend(title='Agency')
plt.show()

# Scatter plot: Response Time vs. Median Travel
plt.figure(figsize=(8, 6))
sns.scatterplot(data=EMS, x='Average Call to Pickup Time', y='Response Time', hue='Responding Agency', palette='viridis')
plt.title('Fig.6 Response(Minutes) vs. Average Call to Pickup Time(Minutes)')
plt.xlabel('Average Call to Pickup Time(Minutes)')
plt.ylabel('Response Time (Minutes)')
plt.legend(title='Agency')
plt.show()

# Select numeric columns
numeric_cols = EMS.select_dtypes(include=['number'])

# Scale numeric columns
EMS_scaled = numeric_cols.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Combine scaled numeric columns with non-numeric columns if needed
non_numeric_cols = EMS.select_dtypes(exclude=['number'])
EMS_scaled = pd.concat([EMS_scaled, non_numeric_cols], axis=1)

# Display the scaled DataFrame
EMS_scaled


from sklearn.cluster import KMeans

#Select only numeric columns for clustering
# Ensure no categorical or non-numeric data is used
numeric_cols = EMS_scaled.select_dtypes(include=['number'])

# Perform KMeans clustering
# Initialize KMeans with the desired number of clusters and a fixed random state
kmeans = KMeans(n_clusters=4, random_state=42)  # n_clusters = 4 for four clusters

# Fit the KMeans model to the numeric data
clustered_EMS_kmeans = kmeans.fit(numeric_cols)

# Assign cluster labels to the DataFrame
# Add a new column 'Cluster' to store cluster labels
EMS_scaled['Cluster'] = clustered_EMS_kmeans.labels_

#Check the data types and filter for numeric columns only
# Ensure that only numeric columns are considered for the summary
numeric_columns = EMS_scaled.select_dtypes(include=['number']).columns

#Group data by clusters and calculate the mean only for numeric columns
cluster_summary = EMS_scaled.groupby('Cluster')[numeric_columns].mean()

# Print the cluster summary to check the mean values for each numeric feature per cluster
print("Cluster Summary:")
print(cluster_summary)

# Step 6: Optional - View the first few rows of the DataFrame with cluster labels
print(EMS_scaled.head())

# Calculate and display centroids of the clusters
# Centroids are stored in the `cluster_centers_` attribute of the fitted KMeans model
centroids = kmeans.cluster_centers_

# Convert centroids to a DataFrame for better readability, using the original numeric column names
import pandas as pd
centroids_df = pd.DataFrame(centroids, columns=numeric_cols.columns)
print("\nCluster Centroids:")
print(centroids_df)

from sklearn.model_selection import train_test_split


# Display basic info about the dataset
print(EMS.info())

# Define features (X) and target (y)
X = EMS.drop(columns=['Response Time'])  # Features
y = EMS['Response Time']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)


print(EMS.dtypes)

import statsmodels.api as sm
import pandas as pd
import numpy as np

# Define response variable
y = EMS["Response Time"]

# Define predictor variables
x = EMS[['Responding Agency', 'Incident Type', 'Average Call to Pickup Time','Number of Incidents', 'Dispatch Time']]

# Create dummy variables for categorical columns and convert them to integers
categorical_columns = ['Responding Agency', 'Incident Type']
x = pd.get_dummies(x, columns=categorical_columns, drop_first=True).astype(int)

# Ensure response variable is numeric
y = pd.to_numeric(y, errors='coerce')

# Drop rows with missing values
data = pd.concat([x, y.rename("Response Time")], axis=1).dropna()

# Separate predictors (x) and response (y)
x = data.drop(columns=["Response Time"])
y = data["Response Time"]

# Add constant to predictors
x = sm.add_constant(x)

# Fit the linear regression model
model = sm.OLS(y, x).fit()

# View model summary
print(model.summary())

# Add constant to predictors again to match the fitted model
x = sm.add_constant(x)

# Fit the linear regression model
model = sm.OLS(y, x).fit()

# Predict values using the model
y_pred = model.predict(x)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((y - y_pred)**2))

# Calculate R-squared
r_squared = model.rsquared

# Output both RMSE and R-squared
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared: {r_squared:.4f}")

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Create dummy variables for categorical columns
categorical_columns = ['Responding Agency', 'Incident Type']
x = pd.get_dummies(EMS[['Responding Agency', 'Incident Type', 'Average Call to Pickup Time',
                        'Number of Incidents', 'Dispatch Time']],
                   columns=categorical_columns, drop_first=True)

# Convert all boolean columns to integers (True -> 1, False -> 0)
x = x.astype(float)

# Define response variable
y = EMS["Response Time"]

# Ensure response variable is numeric
y = pd.to_numeric(y, errors='coerce')

# Drop rows with missing or invalid data
data = pd.concat([x, y.rename("Response Time")], axis=1).dropna()

# Separate predictors (x) and response (y)
x = data.drop(columns=["Response Time"])
y = data["Response Time"]

# Add constant to predictors
x = sm.add_constant(x)

# Fit the OLS model
model = sm.OLS(y, x).fit()

# View model summary
print(model.summary())


from sklearn.metrics import mean_squared_error

# Make predictions using the fitted model
predictions = model.predict(x)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y, predictions))

# Calculate R-squared
r_squared = model.rsquared

# Print the RMSE
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import statsmodels.api as sm
import numpy as np

# Set up cross-validation
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Lists to store RMSE and R-squared values for each fold
rmse_values = []
r2_values = []

# Perform cross-validation
for train_index, test_index in kf.split(x):
    # Split the data
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the OLS model
    model = sm.OLS(y_train, sm.add_constant(x_train)).fit()  # Add constant to the predictors
    
    # Make predictions
    y_pred = model.predict(sm.add_constant(x_test))  # Add constant to the test data
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append(rmse)
    
    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    r2_values.append(r2)

# Calculate the mean RMSE and mean R-squared across all folds, rounded to 4 decimal places
mean_rmse = round(np.mean(rmse_values), 4)
mean_r2 = round(np.mean(r2_values), 4)

# Print results
print(f"Stepwise Forward RMSE: {mean_rmse}")
print(f"Stepwise Forward R-squared: {mean_r2}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Define response variable
y = EMS["Response Time"]

# Define predictor variables
x = EMS[['Responding Agency', 'Incident Type', 'Number of Incidents', 'Average Call to Pickup Time', 
         'Dispatch Time']]

# Create dummy variables for categorical columns and convert them to integers
categorical_columns = ['Responding Agency', 'Incident Type']
x = pd.get_dummies(x, columns=categorical_columns, drop_first=True).astype(int)

# Ensure response variable is numeric
y = pd.to_numeric(y, errors='coerce')

# Drop rows with missing values
data = pd.concat([x, y.rename("Response Time")], axis=1).dropna()

# Separate predictors (x) and response (y)
x = data.drop(columns=["Response Time"])
y = data["Response Time"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12345)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print("Lasso Regression:")
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print(f"Root Mean Squared Error: {rmse_lasso:.4f}")
print(f"R-Squared: {r2_score(y_test, y_pred_lasso):.4f}")


# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print("\nRidge Regression:")
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f"Root Mean Squared Error: {rmse_ridge:.4f}")
print(f"R-Squared: {r2_score(y_test, y_pred_ridge):.4f}")


# ElasticNet Regression
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)
y_pred_elastic = elastic_net.predict(X_test)

print("\nElasticNet Regression:")
rmse_elastic = np.sqrt(mean_squared_error(y_test, y_pred_elastic))
print(f"Root Mean Squared Error: {rmse_elastic:.4f}")
print(f"R-Squared: {r2_score(y_test, y_pred_elastic):.4f}")


import pandas as pd
import statsmodels.api as sm

# Example new data for prediction
new_data = pd.DataFrame({
    'Dispatch Time': [16],  
    'Responding Agency': ['FDNY'],  
    'Number of Incidents': [5],  
    'Incident Type': ['Medical Emergencies'], 
    'Average Call to Pickup Time': [6],  
})

# Create dummy variables for categorical columns in the new data
categorical_columns = ['Responding Agency', 'Incident Type']
new_data = pd.get_dummies(new_data, columns=categorical_columns, drop_first=True)

# Get the training data column names (replace X_train with your actual training DataFrame column names)
train_columns = X_train.columns  # Replace X_train.columns with the actual column names used in the model

# Add missing columns from training set to new_data with default value 0
for col in train_columns:
    if col not in new_data.columns:
        new_data[col] = 0  # Add missing columns as 0

# Ensure the order of columns in new_data matches the order of train_columns
new_data = new_data[train_columns]

# Add constant (intercept) to the new data (as done in the training model)
new_data = sm.add_constant(new_data)

# Make predictions using the fitted model
predictions = model.predict(new_data)

# Print the predictions
print(f"Predictions: {predictions}")

EMS["Responding Agency"].value_counts()

yvar = "Responding Agency"

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Define predictor and target variables
xvars = ["Number Of Incidents", "Response Time"]

# Standardize the predictor variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(EMS[xvars])

# Fit logistic regression with increased max_iter
logreg = LogisticRegression(max_iter=1000, random_state=42)  # Increase max_iter and add random_state for reproducibility
logreg.fit(X_scaled, EMS[yvar])

# Check results
print("Model Coefficients:", logreg.coef_)
print("Model Intercept:", logreg.intercept_)

logreg.predict(X_scaled)

logreg.score(X_scaled, EMS[yvar])

from sklearn.metrics import confusion_matrix
logreg_cm = confusion_matrix(EMS[yvar], logreg.predict(X_scaled))
print(logreg_cm)

logreg_cm/logreg_cm.sum()*100

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(EMS[yvar], logreg.predict(X_scaled))

(logreg_cm[0,0] + logreg_cm[1,1])/logreg_cm.sum()

from sklearn.metrics import precision_score

# Compute precision score for a multiclass target
precision = precision_score(EMS[yvar], logreg.predict(X_scaled), average='weighted')
print("Precision Score:", precision)

from sklearn.metrics import recall_score
recall_score(EMS[yvar], logreg.predict(X_scaled),average='weighted')
logreg_cm[1,1]/(logreg_cm[1,1] + logreg_cm[1,0])

from sklearn.metrics import fbeta_score
fbeta_score(EMS[yvar], logreg.predict(X_scaled), average='weighted', beta = 1)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(EMS[yvar], logreg.predict(X_scaled))

from sklearn.tree import DecisionTreeClassifier
small_tree_classifier = DecisionTreeClassifier(max_depth = 3).fit(X_scaled,EMS[yvar])

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(25,20))
plot_tree(small_tree_classifier, feature_names = xvars, filled = True)

tree_classifier = DecisionTreeClassifier().fit(X_scaled, EMS[yvar]) 

tree_classifier.predict(X_scaled)

tree_classifier.score(X_scaled, EMS[yvar])

logreg.score(X_scaled, EMS[yvar])

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(EMS[yvar], tree_classifier.predict(X_scaled))

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor().fit(X_scaled,EMS["Number Of Incidents"])

from sklearn.metrics import mean_squared_error
tree_reg_rmse = mean_squared_error(EMS["Number Of Incidents"], tree_reg.predict(X_scaled), squared = False)

from sklearn.linear_model import LinearRegression
ols_reg = LinearRegression().fit(X = X_scaled, y = EMS["Number Of Incidents"])
ols_reg_rmse = mean_squared_error(EMS["Number Of Incidents"], ols_reg.predict(X_scaled), squared = False)

print(f"Tree: {tree_reg_rmse:.2f}\nOLS: {ols_reg_rmse:.2f}\n")

EMS.drop(columns = ["Incident Type"], inplace = True)
object_columns = EMS.select_dtypes("O").columns
EMS_randomforest = EMS.join([pd.get_dummies(EMS[cur_col], prefix = cur_col) for cur_col in object_columns])
EMS_randomforest.drop(columns = object_columns, inplace = True)

many_xvars = EMS.columns[EMS.columns != yvar]

from sklearn.ensemble import RandomForestClassifier
forest_classifier = RandomForestClassifier().fit(X = EMS_randomforest[many_xvars], y = EMS[yvar])

forest_classifier.score(EMS_randomforest[many_xvars], EMS[yvar])

from sklearn.model_selection import cross_val_score
num_folds = 5

forest_scores = cross_val_score(
    estimator = RandomForestClassifier(random_state = 872487),
    X = EMS_randomforest[many_xvars],
    y = EMS[yvar],
    cv = num_folds)
forest_scores.mean()

tree_scores = cross_val_score(
    estimator = DecisionTreeClassifier(random_state = 872487),
    X = EMS_randomforest[many_xvars],
    y = EMS[yvar],
    cv = num_folds)
tree_scores.mean()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer

forest_regressor_scores = cross_val_score(
    estimator = RandomForestRegressor(),
    X = EMS[["Response Time", "Response Time"]],
    y = EMS["Number Of Incidents"],
    scoring = make_scorer(mean_squared_error,squared = False),
    cv = num_folds)
forest_regressor_scores.mean()
