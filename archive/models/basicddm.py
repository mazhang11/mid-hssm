import os
import glob
import pandas as pd
import hssm
from utils.preprocessing import load_and_clean_mid_data

def fit_subject_wise_ddm(data_path="../data/mid_data_cleaned_hssm.csv"):
    """
    Loads the cleaned dataset and fits an independent 
    basic DDM for each unique subject.
    """
    print(f"\nLoading cleaned data from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # Get a list of all unique subjects in the dataset
    subjects = df['subject'].unique()
    print(f"Found {len(subjects)} subjects. Starting subject-wise fitting...")
    
    # Storing fitted models in dictionary
    fitted_models = {}
    
    for subject_id in subjects[0:5]:
        print(f"\n" + "="*40)
        print(f" Fitting Basic DDM for Subject: {subject_id}")
        print("="*40)
        
        # 1. Isolate this specific subject's data
        subject_data = df[df['subject'] == subject_id]
        
        # 2. Initialize the standard DDM model in HSSM, no parameter inputs
        my_model = hssm.HSSM(
            data=subject_data,
            model="ddm"
        )
        
        # 3. Fit the model using Bayesian sampling
        my_model.sample(
            tune=1000, 
            draws=1000, 
            chains=2, 
            cores=1
        )
        
        # Store the successfully fitted model
        fitted_models[subject_id] = my_model
        
        print(f"Finished sampling for Subject {subject_id}")
        
    return fitted_models

def run_subject_wise():
    """
    Helper function to run the independent models and output parameter estimates
    for each local test subject.
    """
    models_dict = fit_subject_wise_ddm(data_path="../data/mid_data_cleaned_hssm.csv")
    
    print("\nSUCCESS: All subject-wise models have completed sampling")
    
    first_subject = list(models_dict.keys())[0]
    print(f"\nParameter estimates for Subject {first_subject}:")
    print(models_dict[first_subject].summary())

    second_subject = list(models_dict.keys())[1]
    print(f"\nParameter estimates for Subject {second_subject}:")
    print(models_dict[second_subject].summary())

    third_subject = list(models_dict.keys())[2]
    print(f"\nParameter estimates for Subject {third_subject}:")
    print(models_dict[third_subject].summary())

    fourth_subject = list(models_dict.keys())[3]
    print(f"\nParameter estimates for Subject {fourth_subject}:")
    print(models_dict[fourth_subject].summary())

    fifth_subject = list(models_dict.keys())[4]
    print(f"\nParameter estimates for Subject {fifth_subject}:")
    print(models_dict[fifth_subject].summary())

def fit_hierarchical_basic_ddm(data_path="../data/mid_data_cleaned_hssm.csv"):
    """
    Loads the cleaned dataset and fits a single 
    hierarchical DDM to estimate group-level and subject-level parameters. 
    Currently runs with 5 subjects for local
    """
    print(f"\nLoading cleaned data from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # 1. Isolate the first 5 subjects
    subjects = df['subject'].unique() # get subject list
    subjects_to_keep = subjects[0:5]
    
    print(f"Found {len(subjects)} subjects. Subsetting to first 5: {subjects_to_keep}")
    subset_data = df[df['subject'].isin(subjects_to_keep)].copy()
    
    print("\n" + "="*50)
    print(" Fitting Basic Hierarchical DDM (Group + Individual)")
    print("="*50)
    
    # 2. Initialize the hierarchical model
    my_hierarchical_model = hssm.HSSM(
        data=subset_data,
        model="ddm",
        include=[
            {"name": "v", "formula": "v ~ 1 + (1|subject)"},
            {"name": "a", "formula": "a ~ 1 + (1|subject)"},
            {"name": "t", "formula": "t ~ 1 + (1|subject)"}
            # {"name": "z", "formula": "z ~ 1 + (1|subject)"}
        ]
    )
    
    # 3. Fit the model using Bayesian sampling
    my_hierarchical_model.sample(
        tune=1000, 
        draws=1000, 
        chains=2, 
        cores=1
    )
    
    print("Finished sampling the hierarchical model.")
    
    return my_hierarchical_model

def hierarchical_test():
    """
    Helper function to run the hierarchical model and cleanly print 
    the group-level and subject-level parameters separately.
    """
    hierarchical_model = fit_hierarchical_basic_ddm(data_path="../data/mid_data_cleaned_hssm.csv")
    
    print("\nSUCCESS: Hierarchical model has completed sampling")
    
    # Get the full summary as a Pandas DataFrame
    full_summary = hierarchical_model.summary()
    
    # Separate the group parameters from the subject parameters using text matching.
    # In HSSM's backend, individual subject parameters are placed in brackets 
    # (e.g., 'v_1|subject[PLAJTS027_2]'). The group means are called 'v_Intercept' 
    # and the group variance is 'v_1|subject_sigma'. 
    # We use regex to cleanly separate anything with brackets!
    subject_params = full_summary[full_summary.index.str.contains(r'\[.*\]', regex=True)]
    group_params = full_summary[~full_summary.index.str.contains(r'\[.*\]', regex=True)]
    
    print("\n" + "="*50)
    print(" HIERARCHICAL: GROUP-LEVEL PARAMETERS")
    print("="*50)
    print(group_params)
    
    print("\n" + "="*50)
    print(" HIERARCHICAL: INDIVIDUAL SUBJECT PARAMETERS")
    print("="*50)
    print(subject_params)

def fit_regression_basic_ddm(data_path="../data/mid_data_cleaned_hssm.csv"):
    """
    Loads the cleaned dataset, slices out 5 subjects, and fits a single 
    non-hierarchical DDM where drift rate (v) is predicted by cue_type.
    Formula: v ~ 1 + C(cue_type)
    """
    print(f"\nLoading cleaned data from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # 1. Isolate the first 5 unique subjects
    subjects = df['subject'].unique()
    subjects_to_keep = subjects[0:5]
    
    print(f"Found {len(subjects)} subjects. Subsetting to first 5: {subjects_to_keep}")
    subset_data = df[df['subject'].isin(subjects_to_keep)].copy()
    
    print("\n" + "="*50)
    print(" Fitting Regression DDM: v ~ 1 + C(cue_type) (Non-Hierarchical)")
    print("="*50)
    
    # 2. Initialize the model with a regression formula for v.
    # We specify that 'v' depends on the categorical variable 'cue_type'.
    # The other parameters (a, z, t) will default to a single global estimate.
    my_regression_model = hssm.HSSM(
        data=subset_data,
        model="ddm",
        include=[
            {"name": "v", "formula": "v ~ 1 + C(cue_type)"}
        ]
    )
    
    # 3. Fit the model using Bayesian sampling
    # Keeping tune and draws at 500 for local testing.
    my_regression_model.sample(
        tune=500, 
        draws=500, 
        chains=2, 
        cores=1
    )
    
    print("Finished sampling the basic regression model.")
    
    return my_regression_model

def fit_regression_hierarchical_ddm(data_path="../data/mid_data_cleaned_hssm.csv"):
    """
    Loads the cleaned dataset, slices out 5 subjects, and fits a hierarchical 
    DDM where drift rate (v) is predicted by cue_type, with random effects for subjects.
    Formula: v ~ 1 + C(cue_type) + (1 + C(cue_type)|subject)
    """
    print(f"\nLoading cleaned data from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # 1. Isolate the first 5 unique subjects
    subjects = df['subject'].unique()
    subjects_to_keep = subjects[0:5]
    
    print(f"Found {len(subjects)} subjects. Subsetting to first 5: {subjects_to_keep}")
    subset_data = df[df['subject'].isin(subjects_to_keep)].copy()
    
    print("\n" + "="*50)
    print(" Fitting Regression DDM (Hierarchical)")
    print("="*50)
    
    # 2. Initialize the hierarchical regression model.
    # 'v' gets a fixed effect for cue_type, plus random intercepts and slopes for each subject.
    # We also set 'a', and 't' to vary hierarchically by subject to avoid complete pooling on them.
    # Again, z is intentionally excluded here to avoid dimensionality errors.
    my_hierarchical_regression = hssm.HSSM(
        data=subset_data,
        model="ddm",
        include=[
            {"name": "v", "formula": "v ~ 1 + C(cue_type) + (1 + C(cue_type)|subject)"},
            {"name": "a", "formula": "a ~ 1 + (1|subject)"},
            {"name": "t", "formula": "t ~ 1 + (1|subject)"}
            # {"name": "z", "formula": "z ~ 1 + (1|subject)"}
        ]
    )
    
    # 3. Fit the model using Bayesian sampling, using tune, draws 500 to speed up locally
    my_hierarchical_regression.sample(
        tune=500, 
        draws=500, 
        chains=2, 
        cores=1
    )
    
    print("Finished sampling the hierarchical regression model.")
    
    return my_hierarchical_regression

def run_regression_tests():
    """
    Helper function to run both regression models and print the summaries cleanly.
    Note: Output of individual subject parameters is commented out for the hierarchical
    model to focus only on global parameters.
    """
    # Run the basic regression
    basic_reg_model = fit_regression_basic_ddm()
    print("\nBasic regression model has completed sampling.")
    
    print("\n" + "="*50)
    print(" BASIC REGRESSION PARAMETERS")
    print("="*50)
    print(basic_reg_model.summary())
    
    # Run the hierarchical regression
    hierarchical_reg_model = fit_regression_hierarchical_ddm()
    print("\nHierarchical regression model has completed sampling.")
    
    full_summary = hierarchical_reg_model.summary()
    
    # Separate the group parameters from the subject parameters using regex matching
    subject_params = full_summary[full_summary.index.str.contains(r'\[.*\]', regex=True)]
    group_params = full_summary[~full_summary.index.str.contains(r'\[.*\]', regex=True)]
    
    print("\n" + "="*50)
    print(" HIERARCHICAL REGRESSION: GROUP-LEVEL PARAMETERS")
    print("="*50)
    print(group_params)
    
    print("\n" + "="*50)
    print(" HIERARCHICAL REGRESSION: INDIVIDUAL SUBJECT PARAMETERS")
    print("="*50)
    print(subject_params)

if __name__ == "__main__":
    # 1. Clean data (Will just load the CSV since we already ran it)
    clean_df = load_and_clean_mid_data(
        data_dir="../data", 
        output_filename="mid_data_cleaned_hssm.csv"
    )
    
    # 2. Run the independent subject-wise model fitting
    # print("\n\n>>> STARTING STAGE 1: SUBJECT-WISE MODELING <<<")
    # run_subject_wise()
    
    # 3. Run the hierarchical model fitting
    # print("\n\n>>> STARTING STAGE 2: HIERARCHICAL MODELING <<<")
    # hierarchical_test()
    
    # 4. Run the regression model fitting
    print("\n\n>>> STARTING STAGE 3: REGRESSION MODELING <<<")
    run_regression_tests()