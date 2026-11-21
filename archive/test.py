import hssm

# Load a package-supplied dataset
cav_data = hssm.load_data("cavanagh_theta")

# Define a basic hierarchical model with trial-level covariates
model = hssm.HSSM(
    model="ddm",
    data=cav_data,
    include=[
        {
            "name": "v",
            "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                "theta": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
            },
            "formula": "v ~ theta + (1|participant_id)",
            "link": "identity",
        },
    ],
)

# Sample from the posterior for this model
model.sample()