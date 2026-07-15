import pandas as pd
import numpy as np
import os

def load_stores(file_path):
    """
    Load stores.csv and assign appropriate column names.
    Expected structure: STORE, STORE_NAME, PRICE_ZONE, ZONE_CODE, ZIP, ADDRESS

    Note: ZONE_CODE, ZIP, and ADDRESS may contain null values in the raw data.
    ZONE_CODE and ZIP are loaded as float64 due to NaN presence.
    """
    # stores.csv has no header row
    columns = ['STORE', 'STORE_NAME', 'PRICE_ZONE', 'ZONE_CODE', 'ZIP', 'ADDRESS']
    df = pd.read_csv(file_path, header=None, names=columns)
    
    # Basic type casting and cleaning
    df['STORE'] = df['STORE'].astype(int)
    df['PRICE_ZONE'] = df['PRICE_ZONE'].astype(str).str.strip()
    return df

def load_weeks(file_path):
    """
    Load weeks.csv and assign appropriate column names.
    Expected structure: WEEK, START_DATE, END_DATE, HOLIDAY
    """
    # weeks.csv has no header row
    columns = ['WEEK', 'START_DATE', 'END_DATE', 'HOLIDAY']
    df = pd.read_csv(file_path, header=None, names=columns)
    
    # Basic type casting and cleaning
    df['WEEK'] = df['WEEK'].astype(int)
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], format='%m/%d/%y', errors='coerce')
    df['END_DATE'] = pd.to_datetime(df['END_DATE'], format='%m/%d/%y', errors='coerce')
    df['HOLIDAY'] = df['HOLIDAY'].fillna('').astype(str).str.strip()
    return df

def load_upc(file_path):
    """
    Load UPC.csv containing the product dictionary for cereals.
    Header: COM_CODE,UPC,DESCRIP,SIZE,CASE,NITEM
    """
    df = pd.read_csv(file_path)
    # Basic type casting and cleaning
    df['UPC'] = df['UPC'].astype(np.int64)
    df['DESCRIP'] = df['DESCRIP'].astype(str).str.strip()
    df['SIZE'] = df['SIZE'].astype(str).str.strip()
    return df

def load_movement(file_path):
    """
    Load wcer.csv (weekly movement data). This is a large file (~458 MB).
    Data types are optimized to conserve memory.
    Header: STORE,UPC,WEEK,MOVE,QTY,PRICE,SALE,PROFIT,OK,PRICE_HEX,PROFIT_HEX
    """
    dtypes = {
        'STORE': 'int32',
        'UPC': 'int64',
        'WEEK': 'int32',
        'MOVE': 'int32',
        'QTY': 'int32',
        'PRICE': 'float32',
        'SALE': 'object',
        'PROFIT': 'float32',
        'OK': 'int8'
    }
    # Only load relevant columns (discard HEX representations)
    usecols = ['STORE', 'UPC', 'WEEK', 'MOVE', 'QTY', 'PRICE', 'SALE', 'PROFIT', 'OK']
    df = pd.read_csv(file_path, dtype=dtypes, usecols=usecols)
    return df

def clean_data(df):
    """
    Apply basic cleaning filters:
    1. Filter out economically invalid records: PRICE <= 0 or MOVE < 0 or QTY <= 0.
    2. Drop duplicate records.
    """
    initial_shape = df.shape
    
    # Cleaning filters
    df_clean = df[(df['PRICE'] > 0) & (df['MOVE'] >= 0) & (df['QTY'] > 0)].copy()
    
    # Drop duplicates on (STORE, UPC, WEEK) keys
    df_clean = df_clean.drop_duplicates(subset=['STORE', 'UPC', 'WEEK'], keep='first')
    
    print(f"Cleaning: removed {initial_shape[0] - df_clean.shape[0]} rows out of {initial_shape[0]}.")
    return df_clean

def merge_datasets(movement_df, upc_df, stores_df, weeks_df):
    """
    Merge datasets step-by-step with size validation.
    """
    # 1. Movement + UPC
    m1 = pd.merge(movement_df, upc_df, on='UPC', how='inner')
    print(f"Merge 1 (Movement + UPC): {movement_df.shape} -> {m1.shape}")
    
    # 2. Result + Stores
    m2 = pd.merge(m1, stores_df, on='STORE', how='inner')
    print(f"Merge 2 (with Stores): {m1.shape} -> {m2.shape}")
    
    # 3. Result + Weeks
    final_df = pd.merge(m2, weeks_df, on='WEEK', how='inner')
    print(f"Merge 3 (with Weeks): {m2.shape} -> {final_df.shape}")
    
    return final_df
