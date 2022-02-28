import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Load data

def load_dataset():
    ''' Function to load daatset '''
    df = pd.read_csv('cars.csv')

    # Sanity check
    print(f'#rows= {len(df)} #columns= {len(df.columns)}')
    
    return df



# Function to check NaN or null values in datset
def check_null_values(df):
    '''Do we have NaN in our dataset? '''
    if(df.isnull().values.any() == True):
        print('Dataset has NaN values')
        print(f'Feature/column with NaN values = {df.columns[df.isnull().any()].tolist()}')
        
    return df.columns[df.isnull().any()].tolist()

# Function to impute the missing data
def impute_missing_data(df, feature_list):
    ''' Function to impute null values '''
    for feature in feature_list:
        print(f"mean {feature} = {np.mean(df[feature])}")
        # Impute the missing data wth mean
        df[feature] = df[feature].fillna(df[feature].mean())
        print(f'Imputed feature {feature} with mean values ')
    
    # check, there are no more NaN values in data
    print(f'Dataframe with NaN values = {df.isnull().values.any()}')
    
# Function to encode bool data types/features
def encode_bool_data_type(df):
    ''' Function to encode bool data types '''
    
    # Convert the columns/features of boolean types and map them to
    # True: 1, False: 0
    bool_cols = df.select_dtypes('bool').columns
    for col in bool_cols:
        df[col] = df[col].map({True: 1, False: 0})
        
    return df

# Function to encode features with binary values - having only two uniue values
def encode_binary_data_types(df):
    ''' Functiont to encode features hacing only two unique values '''
    # Note: This function used manual encoding method
    print('Unique values of transmission feature = ', df['transmission'].unique())
    # transmission has only two unique values, so we can encode it as a binary feature.
    transmission_mapping = {'automatic': 0, 'mechanical': 1}

    df['transmission'] = df['transmission'].map(transmission_mapping)
    
    return df

# Function encode features with high cardinality - using labelencoder
def encode_high_cardinality_features(df):
    ''' Function to encode features with high cardinality '''
    
    #Note: This function uses LabelEncoder to encode features
    #      with high cardinality
    
    # Creating an instance of labelencoder
    labelencoder = LabelEncoder()
    
    print('\n #Number of unique items in feature model_name = ', df['model_name'].unique().size)
    
    #model_name has too many unique values, encode them with labelencoder
    df['model_name'] = labelencoder.fit_transform(df['model_name'])
    
    print('\n #Number of unique items in feature manufacturer_name = ', df['manufacturer_name'].unique().size)
    #manufacturer_name has too many unique values, encode them with labelencoder
    df['manufacturer_name'] = labelencoder.fit_transform(df['manufacturer_name'])
    
    return df

# One-hot encoding method
# pandas get_dummies function is the one-hot-encoder
def encode_onehot(_df, f):
    _df2 = pd.get_dummies(_df[f], prefix='', prefix_sep='').groupby(level=0, axis=1).max().add_prefix(f+' - ')
    df3 = pd.concat([_df, _df2], axis=1)
    df3 = df3.drop([f], axis=1)
    
    return df3

if __name__ == "__main__":
    
    print('\n---------------- Load dataset --------------- ')
    
    df = load_dataset()
    print('PASS: Loaded dataset')
    
    
    print('\n---------------- Check for NaN in features --------------- ')
    
    feature_list = check_null_values(df)
    print('PASS: Check for NaN values')
    
    print('\n---------------- Impute null/missing data --------------- ')
    
    impute_missing_data(df, feature_list)
    print('PASS: Impute missing data')
    
    print('\n---------------- Encode bool data type features --------------- ')
    
    df = encode_bool_data_type(df)
    print('PASS: Encoded bool datatype')
    
    print('\n---------------- Encode binary features --------------- ')
    
    df = encode_binary_data_types(df)
    print('PASS: Encoded binary feature type')
    
    print('\n---------------- Encode features having high cardinality ------------- ')
    
    df = encode_high_cardinality_features(df)
    print('\n PASS: Encoded features having high cardinality')
    
    print('\n---------------- One-hot encoding for nominal features ------------------- ')
    # Deep copy original dataframe
    df1 = df.copy()
    # Apply the onehot-encoding method
    df1 = encode_onehot(df1, 'color')
    # Apply the rest of the nominal features too
    df1 = encode_onehot(df1, 'engine_fuel')
    df1 = encode_onehot(df1, 'body_type')
    df1 = encode_onehot(df1, 'state')
    df1 = encode_onehot(df1, 'engine_type')
    df1 = encode_onehot(df1, 'drivetrain')
    
    # Let's check how many features we have
    print(f'Before one-hot encoding = {len(df.columns)}, after one-hot encoding = {len(df1.columns)}')
    print('PASS: One-hot encoded nominal features')
    
    print('\n---------------- Drop the features with irrelavence data -------------------- ')
    
    df1 = df1.drop('location_region', axis=1)
    print('PASS: Irrelavennt features dropped')
    
    print('\n------------------------------------------------------------------------------- ')
    print('Successully pre-processed dataset using various encoding methods')
    print('\n------------------------------------------------------------------------------- ')
    print('**** Exiting **** ')
    
    
    










