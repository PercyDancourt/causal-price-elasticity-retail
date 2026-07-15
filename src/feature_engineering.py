import pandas as pd
import numpy as np

# Module-level constant — import this directly in notebooks to avoid duplication:
#   from feature_engineering import BRAND_MAPPING
BRAND_MAPPING = {
    'KELLOGG': "Kellogg's",
    'KELLOGGS': "Kellogg's",
    'POST': 'Post',
    'GEN': 'General Mills',
    'GENERAL': 'General Mills',
    'QUAKER': 'Quaker',
    'NABISCO': 'Nabisco',
    'RALSTON': 'Ralston'
}

def add_features(df):
    """
    Applies exhaustive feature engineering to the Dominick's Cereal dataset.
    Generates confounders for DML and Causal Forest models:
    - Product Brand (parsed from DESCRIP)
    - Competitor prices (average price of other UPCs in the same store-week, 
      imputed with chain-week average if no competitors are present)
    - Lags of own price and units sold (one-week lag)
    - Lagged price change (log lag price minus log lag 2 price) to capture price dynamics without leaking treatment
    - Specific holiday indicators (Thanksgiving, Christmas, Back-to-school)
    - Continuous time trend (to capture non-linear temporal drift)
    - Interaction between month (seasonality) and store price zone

    Required input columns (must exist in `df` before calling this function):
        - STORE, UPC, WEEK, PRICE: from the merged movement data
        - DESCRIP: product description from UPC lookup (for brand parsing)
        - HOLIDAY: from weeks lookup table (for holiday indicators)
        - PRICE_ZONE: from stores lookup table (for month x zone interaction)
        - units: adjusted units sold (= MOVE / QTY), created in 01_data_preparation
        - month: calendar month extracted from START_DATE, created in 01_data_preparation
    """
    print("Generating engineered features...")
    df_feat = df.copy()
    
    # 0. Brand Taxonomy (extracted from description)
    # Uses module-level BRAND_MAPPING constant (importable by notebooks)
    df_feat['brand'] = df_feat['DESCRIP'].str.split().str[0]
    df_feat['brand'] = df_feat['brand'].map(lambda x: BRAND_MAPPING.get(x, 'Other'))
    
    # 1. Competitor Prices
    # Calculate the average price of all other cereals in the same store and week
    store_week_sum = df_feat.groupby(['STORE', 'WEEK'])['PRICE'].transform('sum')
    store_week_count = df_feat.groupby(['STORE', 'WEEK'])['PRICE'].transform('count')
    
    # Subtract own price to get the average price of competitors
    # When store_week_count <= 1 (no competitors), division yields NaN naturally
    df_feat['competitor_price'] = np.where(
        store_week_count > 1,
        (store_week_sum - df_feat['PRICE']) / (store_week_count - 1),
        np.nan
    )
    
    # Impute missing competitor prices with chain-week average price (average price of all cereals in that week across all stores)
    chain_week_avg = df_feat.groupby('WEEK')['PRICE'].transform('mean')
    df_feat['competitor_price'] = df_feat['competitor_price'].fillna(chain_week_avg)
    df_feat['log_competitor_price'] = np.log(df_feat['competitor_price'] + 1e-5)
    
    # 2. Lags of Own Price and Units (1-week and 2-week lag for price dynamics)
    # Sort values to ensure proper chronological order per store-UPC group
    df_feat = df_feat.sort_values(by=['STORE', 'UPC', 'WEEK']).copy()
    df_feat['lag_price'] = df_feat.groupby(['STORE', 'UPC'])['PRICE'].shift(1)
    df_feat['lag_units'] = df_feat.groupby(['STORE', 'UPC'])['units'].shift(1)
    df_feat['lag_2_price'] = df_feat.groupby(['STORE', 'UPC'])['lag_price'].shift(1)
    
    # Calculate log lags
    df_feat['log_lag_price'] = np.log(df_feat['lag_price'] + 1e-5)
    df_feat['log_lag_units'] = np.log(df_feat['lag_units'] + 1e-5)
    df_feat['log_lag_2_price'] = np.log(df_feat['lag_2_price'] + 1e-5)
    
    # Drop rows where lags are null
    initial_rows = df_feat.shape[0]
    df_feat = df_feat.dropna(subset=['lag_price', 'lag_units', 'lag_2_price']).copy()
    print(f"Dropped {initial_rows - df_feat.shape[0]} rows due to lag missingness.")
    
    # Calculate lagged price change (pre-treatment, no leakage of current PRICE)
    df_feat['lag_price_change'] = df_feat['log_lag_price'] - df_feat['log_lag_2_price']
    
    # 3. Holiday Indicators
    # Specific holiday check from the weeks lookup (Thanksgiving, Christmas)
    df_feat['holiday_thanksgiving'] = df_feat['HOLIDAY'].str.contains('Thanksgiving|Thanks', case=False, na=False).astype(int)
    df_feat['holiday_christmas'] = df_feat['HOLIDAY'].str.contains('Christmas|Xmas', case=False, na=False).astype(int)
    
    # Back-to-school season: August and September
    df_feat['back_to_school'] = df_feat['month'].isin([8, 9]).astype(int)
    
    # 4. Continuous Time Trend
    # Sequential week index from start of dataset
    df_feat['time_trend'] = df_feat['WEEK'].astype(float)
    
    # 5. Month x Zone Interaction Column
    df_feat['month_zone'] = df_feat['month'].astype(str) + "_" + df_feat['PRICE_ZONE'].astype(str)
    
    print("Feature engineering completed successfully.")
    return df_feat
