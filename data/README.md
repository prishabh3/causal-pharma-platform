# MIMIC-III Causal Cohort Data Dictionary

This directory contains pipelines and artifacts for building the causal inference cohort from MIMIC-III data.

## `cohort.parquet` Schema

| Variable | Type | Description |
|----------|------|-------------|
| `HADM_ID` | int | Hospital Admission ID (unique identifier) |
| `age` | float | Age of the patient at admission (years) |
| `treatment` | int | Binary indicator (1=received vasopressors during admission, 0=did not) |
| `outcome` | int | Binary indicator (1=died in hospital within 30 days, 0=survived) |
| `icd_1` to `icd_6` | int | One-hot encoded indicators for the top 6 most common ICD-9 diagnoses |
| `creatinine` | float | Last measured creatinine value within 24h of admission |
| `lactate` | float | Last measured lactate value within 24h of admission |
| `wbc` | float | Last measured white blood cell count within 24h of admission |
| `glucose` | float | Last measured glucose value within 24h of admission |
| `bilirubin` | float | Last measured bilirubin value within 24h of admission |
| `GENDER_M` | int | Binary indicator (1=Male, 0=Female) |
| `ADMISSION_TYPE_*` | int | Binary indicator for admission type (Emergency, Urgent, etc.) |

*Note: Missing continuous labs are median-imputed. Categoricals are mode-imputed.*
