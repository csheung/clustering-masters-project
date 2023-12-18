import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

def import_background_data():
    """
    Function: import raw background data like age, gender, urbanicity, EMS Time Metrics, ROSC Return, Outcome
    Return: a Pandas DataFrame containing the above features
    """
    df = pd.read_csv('processed_data_cardiac_arrest/test_data_region_age_gender_time_outcome.csv')
    return df

def import_csv_data(filepath):
    """
    Function: import relevant feature(s)
    Return: a Pandas DataFrame containing the specific feature(s)
    """
    df = pd.read_csv(filepath)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    return df


# add features to DataFrame
def add_single_feature_into_df(df, file_path, primary_key, mode, col_old_name, col_new_name):
    '''
    Function: merge single feature into the DataFrame
    Return: a combined Pandas DataFrame
    '''
    new_df = import_csv_data(file_path)
    combined_df = pd.merge(df, new_df, on=primary_key, how=mode).rename(columns={col_old_name: col_new_name})
    return combined_df

def add_multiple_features_into_df(df, file_path, primary_key, mode, column_tuple_list):
    '''
    Function: merge multiple features into the DataFrame
    Return: a combined Pandas DataFrame
    '''
    new_df = import_csv_data(file_path)
    combined_df = pd.merge(df, new_df, on=primary_key, how=mode)
    for col_old_name, col_new_name in column_tuple_list:
        combined_df.rename(columns={col_old_name: col_new_name}, inplace=True)
    return combined_df

def add_selected_features_into_df(df, file_path, selected_cols, primary_key, mode, column_tuple_list):
    '''
    Function: merge selected multiple features into the DataFrame
    Return: a combined Pandas DataFrame
    '''
    new_df = hf.import_csv_data(file_path)
    new_df = new_df[selected_cols]
    combined_df = pd.merge(df, new_df, on=primary_key, how=mode)
    for col_old_name, col_new_name in column_tuple_list:
        combined_df.rename(columns={col_old_name: col_new_name}, inplace=True)
    return combined_df


# Manipulate DataFrame
def create_rename_dict(filepath, df, int_type_convert_list, index, column):
    # build dict for column labels
    df = pd.read_csv(filepath)
    
    # convert type to int to get rid of decimals
    if int_type_convert_list:
        hf.convert_type_to_int(df, int_type_convert_list)
        
    # Convert the DataFrame to a dictionary for renaming
    rename_dict = df.set_index(index)[column].to_dict()
    
    return rename_dict

def capitalize_labels_in_rename_dict(dictionary):
    for key, val in dictionary.items(): # inplace modification
        dictionary[key] = ''.join(word.title() for word in val.split())
        
def convert_dict_key_to_str(d):    
    # add new string-typed key into dict
    for k, v in d.items():
        d[str(k)] = v
        
# One-hot Encoding
# Applied in eProcedures
def create_one_hot_encoded_table(df, column_name, rename_dict):

    # One-hot encode the 'eProcedures_03' Airway column
    one_hot_encoded = pd.get_dummies(df[column_name]).astype(int)

    # Rename columns using the provided dictionary
    return one_hot_encoded.rename(columns=rename_dict)
    
def concat_one_hot_encoded_table(df, column_name, rename_dict):
    # Create the one-hot encoded table
    one_hot_encoded = create_one_hot_encoded_table(df, column_name, rename_dict)
    
    # Merge the one-hot encoded columns with the original DataFrame
    combined_df = pd.concat([df, one_hot_encoded], axis=1)
    
    # drop the original column
    combined_df.drop(columns=[column_name], inplace=True)
    
    return combined_df

def group_pcr_key(df, key_column):
    """
    Function: group the repeated keys into unique key
    Return: new DataFrame with unique key for each row
    """
    # Group by 'PcrKey' and sum up the one-hot encoded values for each key
    return df.groupby(key_column).sum().reset_index()

# Applied in eMedications
def create_simple_one_hot_col(df, col, new_col, value_lists):
    """
    function: turn certain values in an existing column into a new one-hot-encoded column.
    return: nothing, modify the passed-in DataFrame in-place.
    """
    df[new_col] = df[col].isin(value_lists).astype(int)
    
# locate the row indexes of selected values
def loc_row_indexes_in_single_col(df, col, value_lists):
    """
    function: locate the row indexes of selected values
    return: a list of Bool, indicating if the value exists
    """
    return df[col].isin(value_lists)

# Create a new column in the df to map the value based on an existing column
def map_values_to_new_col(df, index_mask, col, new_col, value_map):
    """
    function: Create a new column in the df to map the value based on an existing column
    return: nothing, new column added in-place
    """
    df.loc[index_mask, new_col] = df.loc[index_mask, col].map(value_map)
    df[new_col].fillna(0, inplace=True) # fill NaN with 0 <- not in map, no Epinephrine used
    
    
# Calculate the total amount of Epinephrine
def calculate_sum_group_by_pcr_key(df, key, col, new_col):
    """
    function: Calculate the Epinephrine Dosage Total Amount, 
            put them in a new column grouped by primary key
    return: new DataFrame with the new column showing the total amount
    """
    new_df = df.groupby(key)[col].sum().reset_index()
    new_df.rename(columns = {col : new_col}, inplace=True)
    return new_df


# Data type conversion functions
def convert_type_to_int(df, series_name_list):
    """
    Function: turn the dtype for a pandas series to int, coercely remain np.nan
    Return: nothing
    """
    for series_name in series_name_list:
        df[series_name] = pd.to_numeric(df[series_name], errors='coerce').astype('Int64', errors='ignore')
    
def convert_type_to_float(df, series_name_list):
    """
    Function: turn the dtype for a pandas series to float, coercely remain np.nan
    Return: nothing
    """
    for series_name in series_name_list:
        df[series_name] = pd.to_numeric(df[series_name], errors='coerce')
    
def convert_type_to_string(df, series_name_list):
    """
    Function: turn the dtype for a pandas series to string, coercely remain np.nan
    Return: nothing
    """
    for series_name in series_name_list:
        df[series_name] = df[series_name].astype(str, errors='ignore')


# Imputing Data
def sklearn_iter_impute_numeric_cols(curr_df, numeric_impute_cols):
    """
    function: conduct MICE (sklearn iterative imputer) to impute selected numeric columns
    return: df with imputed numerical columns
    """
    # Numeric cols: Initialize the imputer
    numeric_iter_imputer = IterativeImputer(random_state=0, max_iter=100)

    # Impute missing values in numeric columns
    imputed_numeric_data = numeric_iter_imputer.fit_transform(curr_df[numeric_impute_cols])
    imputed_numeric_df = pd.DataFrame(imputed_numeric_data, columns=numeric_impute_cols)
    
    return imputed_numeric_df

# One-hot encoding or label encoding to represent categorical variables as numerical values -> KNN
def knn_impute_categoricals(df, categorical_cols):
    """
    function: conduct KNN Imputation for each categorical column in the array via label encoding
    return: df with imputed categorical columns
    """
    # Store encoders for each column
    encoders = {}

    # Step 1: Encode Categorical Data
    for col in categorical_cols:
        encoder = LabelEncoder()
        
        # Create a mask for non-missing values
        non_missing_mask = df[col].notna()
        
        # Fit and transform non-missing values
        df.loc[non_missing_mask, col] = encoder.fit_transform(df.loc[non_missing_mask, col])
        
        # Store the encoder for later decoding
        encoders[col] = encoder
        
        # Replace np.nan with a placeholder
        # df[col].fillna(-1, inplace=True)

    # Step 2: Apply KNN Imputation
    imputer = KNNImputer()
    df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

    # Step 3: Decode Imputed Values
    for col in categorical_cols:
        encoder = encoders[col]
        df[col] = encoder.inverse_transform(df[col].astype(int))
    
    return df

        

# graph-plotting functions
def plot_continuous_distribution(df, column_name, lower_percentile=5, upper_percentile=95):
    """
    Plot the distribution of a single continuous column in ONE DataFrame.
    
    Parameters:
    - df: DataFrame
    - column_name: Name of the continuous column to be plotted
    - lower_percentile: Lower percentile limit (default is 5)
    - upper_percentile: Upper percentile limit (default is 95)
    """
    # Set the aesthetics for Seaborn plots
    sns.set(style="whitegrid")

    # Ensure the column is continuous
    if df[column_name].dtype in ['int64', 'float64']:
        plt.figure(figsize=(10, 6))
        
        # Calculate the x-axis limits based on the data's percentiles
        lower_limit = df[column_name].quantile(lower_percentile / 100)
        upper_limit = df[column_name].quantile(upper_percentile / 100)
        
        # Plot the distribution with adjusted x-axis limits
        sns.histplot(df[column_name].dropna(), kde=True)
        
        # Set the x-axis limits
        plt.xlim(lower_limit, upper_limit)
        
        plt.title(f'Zoomed Distribution of {column_name}')
        plt.ylabel('Density')
        plt.xlabel(column_name)
        # plt.tight_layout()
        plt.show()
    else:
        print(f"The column '{column_name}' is not continuous.")
        
# plot distribution curves of two DataFrames for comparison
def plot_distribution_comparison(df1, df2, column_name, 
                                         lower_percentile=5, upper_percentile=95,
                                         label1='DataFrame 1', label2='DataFrame 2'):
    """
    Plot the distribution of a single continuous column from two DataFrames.
    
    Parameters:
    - df1, df2: DataFrames to compare
    - column_name: Name of the continuous column to be plotted
    - lower_percentile: Lower percentile limit (default is 5)
    - upper_percentile: Upper percentile limit (default is 95)
    - label1, label2: Labels for the two DataFrames' distributions
    """
    # Set the aesthetics for Seaborn plots
    sns.set(style="whitegrid")

    # Ensure the column is continuous in both DataFrames
    if df1[column_name].dtype in ['int64', 'float64'] and df2[column_name].dtype in ['int64', 'float64']:
        plt.figure(figsize=(10, 6))
        
        # Calculate the combined x-axis limits based on the data's percentiles from both DataFrames
        combined_series = pd.concat([df1[column_name], df2[column_name]])
        lower_limit = combined_series.quantile(lower_percentile / 100)
        upper_limit = combined_series.quantile(upper_percentile / 100)
        
        # Plot the KDE distributions for both DataFrames
        sns.kdeplot(df1[column_name].dropna(), label=label1, shade=True)
        sns.kdeplot(df2[column_name].dropna(), label=label2, shade=True)
        
        # Set the x-axis limits
        plt.xlim(lower_limit, upper_limit)
        
        plt.title(f'Distribution Comparison of {column_name}')
        plt.ylabel('Density')
        plt.xlabel(column_name)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"The column '{column_name}' is not continuous in one or both DataFrames.")

        
# Strip unnecessary spaces before or after the data
def strip_col_str_data(series):
    """
    Function: strip the string-typed data in a Series
    Return: Series with stripped string data
    """
    return series.apply(lambda x: x.strip() if isinstance(x, str) else x)


# Handle unstructured date/time format
def try_parse_date(date_str, date_format):
    """
    Function: Try to parse the date string with the given format
    """
    try:
        datetime.strptime(date_str, date_format)
        return True
    except ValueError:
        return False

def check_datetime_format(series, date_format='%d%b%Y:%H:%M:%S'):
    """
    Function: Check if the datetime string matches the specified format
    Return: Series with True for valid dates and False for invalid/missing dates
    """
    return series.apply(lambda x: True if isinstance(x, str) and 
                        try_parse_date(x, date_format) else False)

