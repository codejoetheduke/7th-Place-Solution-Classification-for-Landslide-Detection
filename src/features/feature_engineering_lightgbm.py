import pandas as pd
import numpy as np
# === Feature Engineering ===
def add_band_indices(df):
    df['B4_B3_ratio'] = df['B4_mean'] / (df['B3_mean'] + 1e-6)
    df['B3_B2_ratio'] = df['B3_mean'] / (df['B2_mean'] + 1e-6)
    df['B2_B1_ratio'] = df['B2_mean'] / (df['B1_mean'] + 1e-6)
    df['B4_B2_ratio'] = df['B4_mean'] / (df['B2_mean'] + 1e-6)
    df['NDI_B4_B3'] = (df['B4_mean'] - df['B3_mean']) / (df['B4_mean'] + df['B3_mean'] + 1e-6)
    df['NDI_B3_B2'] = (df['B3_mean'] - df['B2_mean']) / (df['B3_mean'] + df['B2_mean'] + 1e-6)
    df['NDI_B2_B1'] = (df['B2_mean'] - df['B1_mean']) / (df['B2_mean'] + df['B1_mean'] + 1e-6)
    df['SAR_diff'] = df['B8_mean'] - df['B5_mean']
    return df

def add_stat_features(df):
    for b in range(1, 13):
        df[f'B{b}_range'] = df[f'B{b}_max'] - df[f'B{b}_min']
        df[f'B{b}_iqr'] = df[f'B{b}_q75'] - df[f'B{b}_q25']
        df[f'B{b}_cv'] = df[f'B{b}_std'] / (df[f'B{b}_mean'] + 1e-6)
    return df

def add_texture_features(df):
    for b in range(1, 13):
        df[f'B{b}_texture'] = np.abs(df[f'B{b}_skew']) + np.abs(df[f'B{b}_kurtosis'])
    return df

def add_aggregations(df):
    band_means = [f'B{i}_mean' for i in range(1,13)]
    band_stds = [f'B{i}_std' for i in range(1,13)]
    band_ranges = [f'B{i}_max' for i in range(1,13)]
    df['mean_of_means'] = df[band_means].mean(axis=1)
    df['std_of_means'] = df[band_means].std(axis=1)
    df['sum_of_ranges'] = df[band_ranges].sum(axis=1)
    df['std_of_stds'] = df[band_stds].std(axis=1)
    return df

def engineer_features(df):
    df = add_band_indices(df)
    df = add_stat_features(df)
    df = add_texture_features(df)
    df = add_aggregations(df)
    return df