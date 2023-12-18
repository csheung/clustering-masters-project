import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# Import Data
def import_csv_data(filepath):
    """
    Function: import relevant feature(s)
    Return: a Pandas DataFrame containing the specific feature(s)
    """
    df = pd.read_csv(filepath)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    return df


# One-hot-encoding
def drop_values_from_df(df, col, values_to_drop):
    # Drop the specified values
    return df[~df[col].isin(values_to_drop)]

def one_hot_encode_and_group(df, key_column, categorical_column):
    # One-hot encode the categorical column
    ohe = OneHotEncoder(sparse_output=False, dtype=int)
    
    ohe_df = pd.DataFrame(ohe.fit_transform(df[[categorical_column]]),
                          columns=ohe.get_feature_names_out([categorical_column]))

    # Add the primary key to the one-hot encoded DataFrame
    ohe_df[key_column] = df[key_column].values

    # Group by primary key and use logical or to combine one-hot encoded values
    grouped = ohe_df.groupby(key_column).max().reset_index()
    return grouped


# Data Imputation
def impute_numerical_data_mice(df, numerical_columns):
    imputer = IterativeImputer(max_iter=10, random_state=0)
    
    # Perform imputation
    imputed_data = imputer.fit_transform(df[numerical_columns])
    
    # Ensure no negative values and Age is rounded to the nearest integer
    imputed_data_df = pd.DataFrame(imputed_data, columns=numerical_columns)
    imputed_data_df = imputed_data_df.clip(lower=0)  # Ensure no negative values
    imputed_data_df['Age'] = imputed_data_df['Age'].round().astype(int)  # Round age to nearest integer
    
    return imputed_data_df

def impute_categorical_data(df, categorical_columns):
    # For each categorical column, fill missing values with the most frequent value (mode)
    for column in categorical_columns:
        most_frequent = df[column].mode()[0]  # mode() returns a Series, get the first item
        df[column].fillna(most_frequent, inplace=True)
    return df


# Machine Learning helper functions
# Elbow Method - Find an optimal value for the number of clusters
def elbow(df, inertia):
    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, n_init=10, random_state=0).fit(df)
        inertia.append(kmeans.inertia_)
        



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
