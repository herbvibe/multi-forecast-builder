# District Heating Demand Forecaster

A step-by-step Python project for forecasting district heating demand using real-world data.

## Data Source
- **Flensburg District Heating Network (2020–2024)**
- Source: [Zenodo](https://zenodo.org/records/17177421)
- Hourly heat load and feed flow temperatures for the city of Flensburg, Germany

## Project Structure
```
district-heating-forecaster/
├── data/
│   ├── raw/          # Original downloaded data (do not modify)
│   └── processed/    # Cleaned and transformed data
├── notebooks/        # Jupyter notebooks for exploration and modeling
├── models/           # Saved trained models
├── src/              # Reusable Python modules
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

## Forecast Dashboard

```bash
# From the project root:
streamlit run app.py
```

The dashboard lets you "time-travel" to any date in the 2024 hold-out year and see
what the LightGBM model would have forecast.  Features:
- 48 h demand forecast with 80 % uncertainty band
- Actual vs. forecast overlay for model evaluation
- Weather subplot (temperature + solar radiation)
- Operator event log (maintenance flags, notes)
- Monthly MAPE breakdown + feature importance

## Phases
1. Data loading & exploration
2. Feature engineering
3. Baseline model
4. Advanced forecasting (Prophet / ML)
5. Evaluation & visualization
