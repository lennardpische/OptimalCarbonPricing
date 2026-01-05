# Quantifying the 'Green Premium' in US Domestic Aviation
## A Machine Learning Approach to Modeling Carbon-Adjusted Airfare
### Authors: Lennard Pische, Aram Bagdasarian, Marco Gandola, Vivek Shah

> **Note on Data & Reproducibility:** > The raw source data for this project (DB1B Market & T100 Segment) exceeds 1GB and is not hosted in this repository due to storage limits.  
> 
> **Please view the notebook [`notebooks/OptimalCarbonPricing.ipynb`](notebooks/OptimalCarbonPricing.ipynb) to see the full analysis, pre-rendered visualizations, and model results.**

---

## ✈️ Project Overview
This project investigates the economic relationship between airline pricing and carbon efficiency in the US domestic market. By integrating passenger itinerary data (DB1B) with operational aircraft data (T100), we aim to quantify if a "Green Premium" exists. In other words, do passengers pay more for flights with lower carbon emissions?

## Key Findings
* **Carbon Intensity Variance:** Modeled CO2 emissions per seat-km vary significantly across aircraft types.
* **Price Drivers:** Distance and Carrier dominance remain the primary drivers of airfare, but carbon efficiency shows a measurable interaction effect in specific competitive routes.
* **Model Performance:** The model successfully captures non-linear relationships between flight capacity, load factors, and market fare.

## Methodology

### 1. Data Pipeline (`src/utils.py`)
* **DB1B Market Data:** Aggregated millions of ticket itineraries to calculate passenger-weighted average fares per route.
* **T100 Segment Data:** Extracted payload, fuel burn proxies, and seat configurations to estimate carbon intensity.
* **Feature Engineering:** Created `co2_per_seat_km` and `load_factor` metrics; applied One-Hot Encoding to top aircraft types and major carrier flags.

### 2. Machine Learning
* **Models:** Log-Log Regression with Fixed Effects, Gradient-Boosted Decision Tree, General Additive Model
* **Target:** Average Market Fare ($).
* **Features:** Distance (km), CO2/km, Carrier Market Share, Aircraft Type, Passenger Demand.
* **Validation:** 80/20 Train-Test split with RMSE evaluation.

## Repository Structure
```text
├── notebooks/
│   └── OptimalCarbonPricing.ipynb   # Main analysis with preserved outputs
├── src/
│   └── utils.py                     # Data processing & visualization pipeline
├── images/                          # Exported plots
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation