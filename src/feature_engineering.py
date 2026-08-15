"""Feature engineering: cleaned CSV -> data/processed/diabetic_data_features.csv"""
import pandas as pd

CLEAN_PATH = "data/processed/diabetic_data_clean.csv"
FEATURES_PATH = "data/processed/diabetic_data_features.csv"

AGE_ORDER = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
             '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
AGE_MAP = {age: i for i, age in enumerate(AGE_ORDER)}

MEDICAL_SPECIALTY_MIN_SHARE = 1.0  # percent


def categorize_diagnosis(code) -> str:
    """Group an ICD-9 diag code into one of 9 clinical categories (Strack et al. 2014, Table 2)."""
    if pd.isna(code):
        return 'Missing'
    code = str(code)
    if code.startswith('V') or code.startswith('E'):
        return 'Other'
    try:
        code_num = float(code)
    except ValueError:
        return 'Other'

    if 390 <= code_num <= 459 or code_num == 785:
        return 'Circulatory'
    elif 460 <= code_num <= 519 or code_num == 786:
        return 'Respiratory'
    elif 520 <= code_num <= 579 or code_num == 787:
        return 'Digestive'
    elif 250 <= code_num < 251:
        return 'Diabetes'
    elif 800 <= code_num <= 999:
        return 'Injury'
    elif 710 <= code_num <= 739:
        return 'Musculoskeletal'
    elif 580 <= code_num <= 629 or code_num == 788:
        return 'Genitourinary'
    elif 140 <= code_num <= 239:
        return 'Neoplasms'
    else:
        return 'Other'


def engineer_features(clean_path: str = CLEAN_PATH, output_path: str = FEATURES_PATH) -> pd.DataFrame:
    df_clean = pd.read_csv(clean_path)
    df_features = df_clean.copy()

    # Binary target: any readmission (<30 or >30) vs no readmission
    df_features['readmitted_binary'] = (df_features['readmitted'] != 'NO').astype(int)

    df_features['diag_1_category'] = df_features['diag_1'].apply(categorize_diagnosis)

    # medical_specialty bucketing: keep categories >= 1% of rows, bucket rest into "Other"
    specialty_counts = df_clean['medical_specialty'].value_counts(normalize=True) * 100
    keep_specialties = specialty_counts[specialty_counts >= MEDICAL_SPECIALTY_MIN_SHARE].index.tolist()
    df_features['medical_specialty_grouped'] = df_clean['medical_specialty'].apply(
        lambda x: x if x in keep_specialties else 'Other'
    )

    # Binary race feature: Caucasian vs. Other
    df_features['race_white'] = (df_features['race'] == 'Caucasian').astype(int)

    # Age encoding: convert ordinal age bins to integer (0-9), preserving natural order
    df_features['age_ordinal'] = df_features['age'].map(AGE_MAP)

    # encounter_id/patient_nbr intentionally retained (not dropped) -- excluded from the
    # model's feature set at train time, but needed for Feast entity keys

    df_features.to_csv(output_path, index=False)
    print(f"[engineer_features] Saved {df_features.shape[0]} rows, {df_features.shape[1]} columns -> {output_path}")
    return df_features


if __name__ == "__main__":
    engineer_features()
