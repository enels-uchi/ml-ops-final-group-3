
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, String
from datetime import timedelta

encounter = Entity(name="encounter", join_keys=["encounter_id"])

diabetes_source = FileSource(
    path="data/diabetes_features.parquet",
    timestamp_field="event_timestamp",
)

diabetes_features_v1 = FeatureView(
    name="diabetes_features_v1",
    entities=[encounter],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="time_in_hospital", dtype=Int64),
        Field(name="num_lab_procedures", dtype=Int64),
        Field(name="num_procedures", dtype=Int64),
        Field(name="num_medications", dtype=Int64),
        Field(name="number_outpatient", dtype=Int64),
        Field(name="number_emergency", dtype=Int64),
        Field(name="number_inpatient", dtype=Int64),
        Field(name="number_diagnoses", dtype=Int64),
        Field(name="admission_type_id", dtype=Int64),
        Field(name="discharge_disposition_id", dtype=Int64),
        Field(name="admission_source_id", dtype=Int64),
        Field(name="diag_1_category", dtype=String),
        Field(name="medical_specialty_grouped", dtype=String),
        Field(name="race_white", dtype=Int64),
        Field(name="gender", dtype=String),
        Field(name="age", dtype=String),
    ],
    source=diabetes_source,
)
