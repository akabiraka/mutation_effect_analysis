import sys
sys.path.append("../mutation_effect_analysis")

import pandas as pd

def filter_remove_null_nan_empty_entries(df:pd.DataFrame, a_col_name:str):
    prev_num_rows = df.shape[0]
    df = df[~pd.isna(df[a_col_name])]  # removing nan entries corresponding to a selected col column
    crnt_num_rows = df.shape[0]
    print(f"Number of NAN rows removed: {prev_num_rows-crnt_num_rows}")
    
    prev_num_rows = df.shape[0]    
    df = df[~pd.isnull(df[a_col_name])]  # removing null entries corresponding to a selected col column
    crnt_num_rows = df.shape[0]
    print(f"Number of NULL rows removed: {prev_num_rows-crnt_num_rows}")
    
    prev_num_rows = df.shape[0]
    df = df[df[a_col_name] != ""]
    crnt_num_rows = df.shape[0]
    print(f"Number of empty rows removed: {prev_num_rows-crnt_num_rows}")
    return df
