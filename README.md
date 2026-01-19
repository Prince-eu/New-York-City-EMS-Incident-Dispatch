# New York City EMS Incident Dispatch Analysis

## Description

This project applies advanced data science and machine learning techniques to analyze New York City's 911 emergency response system, one of the most active emergency response networks in the country handling over 9 million calls annually. Using a comprehensive dataset of emergency incidents spanning from November 2013 to November 2024, this analysis examines response times, dispatch efficiency, and operational patterns across three major emergency agencies: EMS (Emergency Medical Services), FDNY (Fire Department of New York), and NYPD (New York Police Department). Through exploratory data analysis, feature engineering, and predictive modeling—including Decision Trees, Random Forests, and K-Nearest Neighbors (KNN)—the study identifies key factors affecting response times and provides data-driven insights to optimize emergency response operations. By transforming raw incident data into actionable intelligence, this project supports evidence-based decision-making to enhance public safety outcomes and improve the efficiency of life-saving emergency services.

## Project Goals

- Analyze response time patterns across EMS, FDNY, and NYPD to identify operational inefficiencies and bottlenecks
- Predict emergency response times using machine learning models to support resource allocation and planning
- Compare agency performance across different incident types and identify areas for improvement
- Develop actionable recommendations for optimizing dispatch workflows and reducing response delays
- Extract insights from temporal patterns (time of day, day of week, seasonal trends) affecting emergency response efficiency

## Business Problem

Despite the critical importance of rapid emergency response, significant inconsistencies in response times among EMS, FDNY, and NYPD reveal potential inefficiencies in the current emergency dispatch system. The data indicates substantial variations in response metrics for life-threatening medical emergencies compared to non-critical incidents, as well as notable discrepancies across agencies responding to similar emergency types.

These inefficiencies may result from:
- Resource constraints and suboptimal allocation patterns
- Poor coordination between multiple responding agencies
- Logistical bottlenecks in dispatch and travel workflows
- Variations in incident prioritization and categorization

Factors such as incident type, call-to-dispatch rates, and travel durations may significantly hinder the ability of emergency agencies to respond effectively. Without comprehensive analysis and optimization of response workflows, these delays will persist, potentially resulting in preventable loss of life and property. This project addresses these critical inefficiencies by identifying key predictors of response times and developing predictive models to support operational improvements.

## Data and Scope

**Dataset:** 911 End-to-End Data  
**Source:** City of New York Mayor's Office of Operations  
**Time Period:** November 2013 - November 2024  
**Total Records:** 10,508 reported incidents  
**Update Frequency:** Monthly (collected on a week-on-week basis per agency and incident type)  

**Dataset Structure:**
- **30 variables** including quantitative, continuous, and categorical predictors
- **Primary outcome variable:** "Response Time - Call to Agency Arrival" (time from call receipt to on-scene arrival)
- **Key predictors:** Average Call to Pickup Time, Average Travel Time, Average Dispatch Time, Incident Type, Responding Agency, Time of Day, Day of Week

**Data Source:** [911 End-to-End Data - Data.gov](https://catalog.data.gov/dataset/911-end-to-end-data)

## Technologies Used

- **Python 3.x** - Primary programming language
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **scikit-learn** - Machine learning algorithms including:
  - Decision Tree Classifier/Regressor
  - Random Forest Classifier/Regressor
  - K-Nearest Neighbors (KNN)
  - Cross-validation and model evaluation
- **Deepnote** - Collaborative data science notebook environment

## Files in this Repository

- **`911_End-to-End_Data.csv`** - Primary dataset containing 10,508 emergency incident records with response metrics
- **`911_E2E_Data_Dictionary.xlsx`** - Comprehensive data dictionary explaining all 30 variables and their definitions
- **`EMS_incident_dispatch_data_description.xlsx`** - Additional metadata and descriptive statistics for the dataset
- **`EMS_Incident_Dispatch.ipynb`** - Main Jupyter notebook containing the complete analysis workflow:
  - Data loading and preprocessing
  - Exploratory data analysis (EDA)
  - Feature engineering
  - Machine learning model development (Decision Trees, Random Forests, KNN)
  - Model evaluation and cross-validation
  - Results visualization and interpretation
- **`EMS_Incident_Dispatch.py`** - Python script version of the analysis for reproducibility and automation
- **`README.md`** - This file, providing project overview and documentation

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Analysis of response time distributions across agencies and incident types
- Temporal pattern identification (hourly, daily, weekly, seasonal trends)
- Correlation analysis between predictor variables and response times
- Identification of outliers and anomalies in emergency response data

### 2. Data Preprocessing & Feature Engineering
- Handling missing values and data quality issues
- Creation of derived features (e.g., time of day categories, incident urgency levels)
- Categorical encoding for machine learning models
- Feature scaling and normalization where appropriate

### 3. Predictive Modeling

**Classification Models** (predicting incident categories and response priorities):
- **Decision Tree Classifier** - Baseline model for interpretability
- **Random Forest Classifier** - Ensemble model achieving **99.97% accuracy**
- Cross-validation with multiple folds for robust performance evaluation

**Regression Models** (predicting continuous response times):
- **Random Forest Regressor** - Predicting number of incidents and response times
- Performance measured using Root Mean Squared Error (RMSE)
- Achieved RMSE of approximately 2067 incidents (prediction error metric)

**Model Comparison:**
- Random Forest Classifier: 99.97% accuracy
- Decision Tree Classifier: 99.82% accuracy
- Random Forest demonstrates superior performance over single decision trees

### 4. Model Evaluation
- Cross-validation scoring to prevent overfitting
- Comparison of multiple algorithms to identify best-performing models
- Feature importance analysis to identify key predictors of response times

## Key Findings

1. **Model Performance Excellence:** Random Forest models achieved exceptional predictive accuracy (99.97%), demonstrating the feasibility of using machine learning to predict emergency response patterns and optimize resource allocation.

2. **Agency-Specific Patterns:** Significant variations exist in response times across EMS, FDNY, and NYPD for similar incident types, suggesting opportunities for cross-agency coordination improvements.

3. **Incident Type Impact:** Life-threatening emergencies (e.g., cardiac arrests, major fires) show different response patterns compared to non-critical incidents, validating current prioritization systems but revealing potential for optimization.

4. **Temporal Trends:** Response times vary significantly by time of day, day of week, and season, indicating the need for dynamic resource allocation strategies that adapt to predictable demand patterns.

5. **Dispatch Efficiency:** Call-to-dispatch time and travel time are critical predictors of overall response time, highlighting these as key leverage points for operational improvements.

6. **Predictive Capability:** The high accuracy of machine learning models demonstrates that response times can be reliably predicted based on incident characteristics, enabling proactive resource planning and dispatch optimization.

## Actionable Recommendations

1. **Implement predictive dispatch systems** using machine learning to anticipate high-demand periods and pre-position resources
2. **Optimize inter-agency coordination** by identifying and addressing response time discrepancies between agencies
3. **Enhance resource allocation** during peak demand hours identified through temporal analysis
4. **Reduce dispatch delays** by streamlining call processing and routing procedures
5. **Conduct targeted training** for dispatch operators based on incident type patterns that show prolonged response times

## Future Work

- Incorporate real-time data feeds for dynamic response time prediction
- Develop geospatial analysis to identify high-demand geographic areas
- Integrate weather data and special events to improve prediction accuracy
- Build interactive dashboards for real-time emergency dispatch monitoring
- Expand analysis to include patient outcomes and incident resolution metrics

## Contributors

-  Prince Enyiorji
-  Tsungai Tsambatare
-  Sharon Paruwani
