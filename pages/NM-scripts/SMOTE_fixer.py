import pandas as pd
import io

# Define input and output file paths
input_smote_file_path = '../../Data/NM-datasets/Breast_Cancer_train_smote.csv'
output_fixed_file_path = '../../Data/NM-datasets/Breast_Cancer_train_smote_fixed.csv'

# Load the SMOTE dataset from the specified file
try:
    df_smote = pd.read_csv(input_smote_file_path)
except FileNotFoundError:
    print(f"Error: Input SMOTE file not found at {input_smote_file_path}")
    exit()
except Exception as e:
    print(f"Error reading SMOTE file: {e}")
    exit()


# Clean up column names from potential leading/trailing spaces if read from CSV
df_smote.columns = df_smote.columns.str.strip()


# Define the target columns in the desired order for the original dataset
og_columns_ordered = [
    'Age', 'Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage',
    'differentiate', 'Grade', 'A Stage', 'Tumor Size', 'Estrogen Status',
    'Progesterone Status', 'Regional Node Examined', 'Reginol Node Positive',
    'Survival Months', 'Status'
]

df_og = pd.DataFrame()


# --- Transformation Stuff ---

# Make sure these columns exist in df_smote before trying to assign them
direct_cols_to_check = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months', 'Status']
for col in direct_cols_to_check:
    if col not in df_smote.columns:
        print(f"Error: Expected direct column '{col}' not found in SMOTE data.")
        exit()
df_og['Age'] = df_smote['Age']
df_og['Tumor Size'] = df_smote['Tumor Size']
df_og['Regional Node Examined'] = df_smote['Regional Node Examined']
df_og['Reginol Node Positive'] = df_smote['Reginol Node Positive']
df_og['Survival Months'] = df_smote['Survival Months']
df_og['Status'] = df_smote['Status']

# Helper to check if all necessary one-hot columns exist for a group
def check_ohe_cols(df, cols_list, group_name):
    for col in cols_list:
        if col not in df.columns:
            print(f"Error: Expected one-hot encoded column '{col}' for group '{group_name}' not found in SMOTE data.")
            print(f"Available columns: {list(df.columns)}")
            exit()
    return True

# Race
race_ohe_cols = ['Race_White', 'Race_Other']
check_ohe_cols(df_smote, race_ohe_cols, 'Race')
def determine_race(row):
    if row['Race_White']:
        return 'White'
    elif row['Race_Other']:
        return 'Other'
    return 'Unknown'
df_og['Race'] = df_smote.apply(determine_race, axis=1)

# Marital Status
marital_ohe_cols = ['Marital Status_Married', 'Marital Status_Separated', 'Marital Status_Single', 'Marital Status_Widowed']
check_ohe_cols(df_smote, marital_ohe_cols, 'Marital Status')
def determine_marital_status(row):
    if row['Marital Status_Married']:
        return 'Married'
    elif row['Marital Status_Separated']:
        return 'Separated'
    elif row['Marital Status_Single']:
        return 'Single'
    elif row['Marital Status_Widowed']:
        return 'Widowed'
    return 'Divorced' # Base category
df_og['Marital Status'] = df_smote.apply(determine_marital_status, axis=1)

# T Stage
tstage_ohe_cols = ['T Stage _T2', 'T Stage _T3', 'T Stage _T4']
check_ohe_cols(df_smote, tstage_ohe_cols, 'T Stage')
def determine_t_stage(row):
    if row['T Stage _T2']:
        return 'T2'
    elif row['T Stage _T3']:
        return 'T3'
    elif row['T Stage _T4']:
        return 'T4'
    return 'T1' # Base category
df_og['T Stage'] = df_smote.apply(determine_t_stage, axis=1)

# N Stage
nstage_ohe_cols = ['N Stage_N2', 'N Stage_N3']
check_ohe_cols(df_smote, nstage_ohe_cols, 'N Stage')
def determine_n_stage(row):
    if row['N Stage_N2']:
        return 'N2'
    elif row['N Stage_N3']:
        return 'N3'
    return 'N1' # Base category
df_og['N Stage'] = df_smote.apply(determine_n_stage, axis=1)


# 6th Stage
sixth_stage_ohe_cols = ['6th Stage_IIB', '6th Stage_IIIA', '6th Stage_IIIB', '6th Stage_IIIC']
check_ohe_cols(df_smote, sixth_stage_ohe_cols, '6th Stage')
def determine_6th_stage(row):
    if row['6th Stage_IIB']:
        return 'IIB'
    elif row['6th Stage_IIIA']:
        return 'IIIA'
    elif row['6th Stage_IIIB']:
        return 'IIIB'
    elif row['6th Stage_IIIC']:
        return 'IIIC'
    return 'IIA' # Base category
df_og['6th Stage'] = df_smote.apply(determine_6th_stage, axis=1)

# Differentiate
differentiate_ohe_cols = ['differentiate_Poorly differentiated', 'differentiate_Undifferentiated', 'differentiate_Well differentiated']
check_ohe_cols(df_smote, differentiate_ohe_cols, 'differentiate')
def determine_differentiate(row):
    if row['differentiate_Poorly differentiated']:
        return 'Poorly differentiated'
    elif row['differentiate_Undifferentiated']:
        return 'Undifferentiated'
    elif row['differentiate_Well differentiated']:
        return 'Well differentiated'
    return 'Moderately differentiated' # Base category
df_og['differentiate'] = df_smote.apply(determine_differentiate, axis=1)

# Grade
grade_ohe_cols = ['Grade_1', 'Grade_2', 'Grade_3']
check_ohe_cols(df_smote, grade_ohe_cols, 'Grade')
def determine_grade(row):
    if row['Grade_1']:
        return '1'
    elif row['Grade_2']:
        return '2'
    elif row['Grade_3']:
        return '3'
    return 'Unknown' # Just in case...
df_og['Grade'] = df_smote.apply(determine_grade, axis=1)

# A Stage
astage_ohe_cols = ['A Stage_Regional']
check_ohe_cols(df_smote, astage_ohe_cols, 'A Stage')
def determine_a_stage(row):
    if row['A Stage_Regional']:
        return 'Regional'
    return 'Distant' # Assuming 'Distant' as the base category if not 'Regional'
df_og['A Stage'] = df_smote.apply(determine_a_stage, axis=1)

# Estrogen Status
estrogen_ohe_cols = ['Estrogen Status_Positive']
check_ohe_cols(df_smote, estrogen_ohe_cols, 'Estrogen Status')
def determine_estrogen_status(row):
    if row['Estrogen Status_Positive']:
        return 'Positive'
    return 'Negative' # Base category
df_og['Estrogen Status'] = df_smote.apply(determine_estrogen_status, axis=1)

# Progesterone Status
progesterone_ohe_cols = ['Progesterone Status_Positive']
check_ohe_cols(df_smote, progesterone_ohe_cols, 'Progesterone Status')
def determine_progesterone_status(row):
    if row['Progesterone Status_Positive']:
        return 'Positive'
    return 'Negative' # Base category
df_og['Progesterone Status'] = df_smote.apply(determine_progesterone_status, axis=1)


# Ensure the final DataFrame has columns in the specified OG order
df_og_final = df_og[og_columns_ordered]

# Display the result
print("--- Original Format Dataset (First 5 rows) ---")
print(df_og_final.head().to_string())

# Save to a CSV file
try:
    df_og_final.to_csv(output_fixed_file_path, index=False)
    print(f"\nSuccessfully saved reformatted data to: {output_fixed_file_path}")
except Exception as e:
    print(f"\nError saving output file: {e}")