import pandas as pd
import hssm
import arviz as az
import matplotlib.pyplot as plt

def prepare_continuous_covariates(data_path="../data/mid_data_cleaned_hssm.csv"):
    """
    Loads the cleaned data and maps the categorical cue types 
    to a continuous numerical scale for regression modeling.
    """
    print(f"Loading cleaned data from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # Map categorical cue strings to continuous numerical values
    cue_mapping = {
        'NoIncentive': 0.0,
        'Quarters': 0.5,
        'Dollar': 1.0,
        'FiveDollars': 5.0
    }
    
    df['cue_value'] = df['cue_type'].map(cue_mapping)
    print("Successfully mapped continuous covariates.")
    
    return df

def fit_categorical_model(df):
    """
    Fits a hierarchical DDM treating cue_type as distinct categories.
    Bias (z) is omitted, defaulting to a fixed 0.5.
    """
    # --- MODIFIED FOR OSCAR: Removed [0:5] to use the full dataset ---
    # Isolate the first 5 subjects for local testing (Comment retained)
    # subjects = df['subject'].unique()[0:5]
    # subset_data = df[df['subject'].isin(subjects)].copy()
    subset_data = df.copy() # Using full dataset for OSCAR
    
    print("\n" + "="*50)
    # --- MODIFIED FOR OSCAR: Updated print statement ---
    print(" Fitting Categorical Regression Model (Full Dataset)")
    print("="*50)
    
    # Initialize the categorical model
    # C(cue_type) forces the model to estimate distinct shifts for each category
    categorical_model = hssm.HSSM(
        data=subset_data,
        model="ddm",
        include=[
            {"name": "v", "formula": "v ~ 1 + C(cue_type) + (1 + C(cue_type)|subject)"},
            {"name": "a", "formula": "a ~ 1 + (1|subject)"},
            {"name": "t", "formula": "t ~ 1 + (1|subject)"}
            # z is omitted, so it defaults to fixed 0.5 and will not update
        ]
    )
    
    # --- MODIFIED FOR OSCAR: Increased chains and cores to 4 ---
    categorical_model.sample(tune=1000, draws=1000, chains=4, cores=4)
    print("Finished sampling categorical model.")
    return categorical_model

def fit_continuous_model(df):
    """
    Fits a hierarchical DDM treating the incentive as a continuous linear predictor.
    Bias (z) is omitted, defaulting to a fixed 0.5.
    """
    # --- MODIFIED FOR OSCAR: Removed [0:5] to use the full dataset ---
    # Isolate the first 5 subjects for local testing (Comment retained)
    # subjects = df['subject'].unique()[0:5]
    # subset_data = df[df['subject'].isin(subjects)].copy()
    subset_data = df.copy() # Using full dataset for OSCAR
    
    print("\n" + "="*50)
    # --- MODIFIED FOR OSCAR: Updated print statement ---
    print(" Fitting Continuous Regression Model (Full Dataset)")
    print("="*50)
    
    # Initialize the continuous model
    # 'cue_value' is numerical, so the model calculates a single slope (beta weight)
    # for how much v increases per $1 increase in reward.
    continuous_model = hssm.HSSM(
        data=subset_data,
        model="ddm",
        include=[
            {"name": "v", "formula": "v ~ 1 + cue_value + (1 + cue_value|subject)"},
            {"name": "a", "formula": "a ~ 1 + (1|subject)"},
            {"name": "t", "formula": "t ~ 1 + (1|subject)"}
            # z is omitted, so it defaults to fixed 0.5 and will not update
        ]
    )
    
    # --- MODIFIED FOR OSCAR: Increased chains and cores to 4 ---
    continuous_model.sample(tune=1000, draws=1000, chains=4, cores=4)
    print("Finished sampling continuous model.")
    return continuous_model

def plot_model_posteriors(model, model_name="Model"):
    """
    Uses ArviZ to plot marginal posteriors and pair plots 
    to visually inspect parameter estimates and tradeoffs.
    """
    print(f"\nGenerating plots for {model_name}...")
    
    # --- MODIFIED FOR OSCAR: Added prefix generation for file saving ---
    file_prefix = model_name.replace(" ", "_")
    
    # 1. Plot the marginal posteriors for the global intercepts
    # This shows the confidence range for the group average of each parameter
    az.plot_posterior(
        model.traces, 
        var_names=['v_Intercept', 'a_Intercept', 't_Intercept']
    )
    plt.suptitle(f"{model_name}: Marginal Posteriors")
    
    # --- MODIFIED FOR OSCAR: Swapped plt.show() for plt.savefig() and plt.close() ---
    # plt.show()
    plt.savefig(f"{file_prefix}_marginals.png", dpi=300, bbox_inches='tight')
    plt.close() # Close the plot so it doesn't take up headless memory
    
    # 2. Plot pair plots to check for parameter tradeoffs
    # kind='kde' creates a contour plot. A strong diagonal stretch means high tradeoff.
    az.plot_pair(
        model.traces, 
        var_names=['v_Intercept', 'a_Intercept'], 
        kind='kde', 
        marginals=True
    )
    plt.suptitle(f"{model_name}: Tradeoff Check (v vs a)")
    
    # --- MODIFIED FOR OSCAR: Swapped plt.show() for plt.savefig() and plt.close() ---
    # plt.show()
    plt.savefig(f"{file_prefix}_tradeoffs.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots for {model_name}.")

if __name__ == "__main__":
    # 1. Load data and create the continuous covariate column
    df = prepare_continuous_covariates()
    
    # 2. Fit Categorical Model
    # categorical_model = fit_categorical_model(df)
    # plot_model_posteriors(categorical_model, "Categorical Model")
    
    # 3. Fit Continuous Model
    continuous_model = fit_continuous_model(df)
    plot_model_posteriors(continuous_model, "Continuous Model")