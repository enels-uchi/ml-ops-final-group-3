"""Data cleaning: raw diabetic_data.csv -> data/processed/diabetic_data_clean.csv"""
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/diabetic_data.csv"
CLEAN_PATH = "data/processed/diabetic_data_clean.csv"

HOSPICE_EXPIRED_CODES = [11, 13, 14, 19, 20]


def clean_data(raw_path: str = RAW_PATH, output_path: str = CLEAN_PATH) -> pd.DataFrame:
    df_raw = pd.read_csv(raw_path)
    df_raw = df_raw.replace('?', np.nan)

    df_clean = df_raw.copy()

    # Drop high-missingness, low-relevance columns
    df_clean = df_clean.drop(columns=['weight', 'payer_code'])

    # Fill missing categoricals
    df_clean['race'] = df_clean['race'].fillna('Missing')
    df_clean['medical_specialty'] = df_clean['medical_specialty'].fillna('Missing')

    # Recode disguised missing-value codes in ID columns to a unified "Missing" marker (-1)
    df_clean['admission_type_id'] = df_clean['admission_type_id'].replace([5, 6, 8], -1)
    df_clean['discharge_disposition_id'] = df_clean['discharge_disposition_id'].replace([18, 25], -1)
    df_clean['admission_source_id'] = df_clean['admission_source_id'].replace([9, 17, 20], -1)

    # Remove hospice/expired encounters (readmission undefined for these patients)
    df_clean = df_clean[~df_clean['discharge_disposition_id'].isin(HOSPICE_EXPIRED_CODES)]

    # Remove rows with invalid gender
    df_clean = df_clean[df_clean['gender'] != 'Unknown/Invalid']

    # Keep only the first encounter per patient (prevents train/test leakage)
    df_clean = df_clean.sort_values('encounter_id').drop_duplicates(subset='patient_nbr', keep='first')

    df_clean.to_csv(output_path, index=False)
    print(f"[clean_data] Saved {df_clean.shape[0]} rows, {df_clean.shape[1]} columns -> {output_path}")
    return df_clean


if __name__ == "__main__":
    clean_data()
