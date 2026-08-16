"""Pure-pandas replay of the AutoML champion's PyCaret preprocessing.

The champion pipeline (`models/model_automl.pkl`) preprocesses the 16 raw features into the 35
columns CatBoost was fitted on. Rather than install PyCaret + scikit-learn 1.4 +
category_encoders in the serving image just to run four transforms, `src/export_champion.py`
extracts those transforms' fitted parameters to JSON and this module replays them with pandas
alone.

The four stages, in the pipeline's own order:

  1. numeric imputation      — SimpleImputer(strategy="mean"), per-column mean from training
  2. categorical imputation  — SimpleImputer(strategy="most_frequent")
  3. ordinal encoding        — `gender` -> integer code; unseen category -> -1
  4. one-hot encoding        — `diag_1_category`, `medical_specialty_grouped`; unseen category
                               yields an all-zero block, matching handle_unknown="value"

Column order is not re-derived here — it is replayed verbatim from the exported
`output_columns`, because CatBoost is fitted on positions as well as names.

Equivalence with the original pipeline is enforced at export time: `export_champion.py` refuses
to write the artifact unless this replay reproduces `pipeline.predict_proba` to within 1e-9 over
the full dataset.
"""
import json
from pathlib import Path

import pandas as pd

UNKNOWN_ORDINAL = -1  # category_encoders' handle_unknown="value" code for unseen categories


class ChampionPreprocessor:
    """Replays the champion pipeline's transform from exported parameters."""

    def __init__(self, spec: dict):
        self.numeric_columns = spec["numeric_columns"]
        self.numeric_fill = spec["numeric_fill"]
        self.categorical_columns = spec["categorical_columns"]
        self.categorical_fill = spec["categorical_fill"]
        self.ordinal_column = spec["ordinal_column"]
        self.ordinal_map = spec["ordinal_map"]
        self.onehot = spec["onehot"]
        self.output_columns = spec["output_columns"]

    @classmethod
    def from_dir(cls, directory: str) -> "ChampionPreprocessor":
        with open(Path(directory) / "preprocessing.json") as f:
            return cls(json.load(f))

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()

        # 1. numeric imputation. Coercing first matters: a column absent from every record in a
        #    batch arrives as all-NA object dtype, which would otherwise survive as non-numeric.
        for col in self.numeric_columns:
            series = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(
                index=df.index, dtype="float64"
            )
            df[col] = series.fillna(self.numeric_fill[col])

        # 2. categorical imputation
        for col in self.categorical_columns:
            series = df[col] if col in df else pd.Series(index=df.index, dtype="object")
            df[col] = series.astype("object").where(series.notna(), self.categorical_fill[col])

        # 3. ordinal encoding of `gender`
        df[self.ordinal_column] = (
            df[self.ordinal_column].astype(str).map(self.ordinal_map).fillna(UNKNOWN_ORDINAL)
        ).astype("int64")

        # 4. one-hot expansion. Built explicitly rather than via get_dummies so that unseen
        #    categories produce an all-zero block instead of a surprise extra column.
        for col, categories in self.onehot.items():
            values = df[col].astype(str)
            for category in categories:
                df[f"{col}_{category}"] = (values == category).astype("int64")
            df = df.drop(columns=[col])

        # Replay the fitted column order exactly; anything unexpected is dropped, anything
        # missing is filled with 0.
        return df.reindex(columns=self.output_columns, fill_value=0)
