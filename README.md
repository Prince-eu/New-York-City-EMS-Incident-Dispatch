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
- **scikit-learn** - Machine learning and statistical modeling:
  - Linear Regression (OLS)
  - Lasso, Ridge, and Elastic Net regularization
  - Cross-validation (10-fold CV)
  - Model evaluation metrics (RMSE, R-squared)
  - Logistic Regression (classification)
  - Decision Trees and Random Forests (exploratory analysis)
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
- Creation of derived features for temporal analysis
- Variable selection and multicollinearity assessment
- Feature scaling and normalization where appropriate

### 3. Supervised Machine Learning - Regression Models

**Primary Analysis: Linear Regression Approaches**

1. **Ordinary Least Squares (OLS) Regression**
   - Baseline model establishing linear relationships between predictors and response times
   - Identification of significant predictors and their coefficients

2. **Stepwise Forward Regression**
   - Sequential variable selection to identify the most predictive features
   - Building models incrementally based on statistical significance

3. **Stepwise RMSE-Based Selection**
   - Variable selection optimized for Root Mean Squared Error minimization
   - Focus on predictive accuracy rather than statistical significance alone

**Advanced Regression with 10-Fold Cross-Validation:**

4. **Regularized Regression Models**
   - **Lasso Regression** - L1 regularization for feature selection and sparsity
   - **Ridge Regression** - L2 regularization to handle multicollinearity
   - **Elastic Net** - Combined L1/L2 regularization balancing feature selection and stability
   - Cross-validation used to optimize regularization parameters and prevent overfitting

5. **Predictive Modeling**
   - Using optimized models to make predictions on emergency response times
   - Model evaluation based on RMSE and cross-validated performance metrics

### 4. Appendix: Exploratory Non-Linear Methods

**Additional techniques explored to assess non-linear patterns:**

- **Logistic Regression** - Classification of incident priority levels
- **Decision Trees** - Tree-based models for interpretable non-linear patterns
- **Tree-based prediction** - Fitting Number of Incidents using decision trees
- **Random Forests** - Ensemble methods for both classification and regression
  - Random Forest Regressor with RMSE of approximately 2067 for incident prediction

### 5. Model Evaluation
- 10-fold cross-validation for robust performance estimation
- RMSE as primary metric for regression model comparison
- Feature importance analysis to identify key predictors of response times
- Comparison across linear and non-linear approaches

## Key Findings

1. **Regularized Regression Performance:** Lasso, Ridge, and Elastic Net models with 10-fold cross-validation provided robust predictions of emergency response times, with regularization effectively handling multicollinearity and preventing overfitting in the presence of numerous predictors.

2. **Stepwise Regression Insights:** RMSE-based stepwise selection identified the most predictive features for response times, balancing model complexity with predictive accuracy and revealing which factors most significantly impact emergency response efficiency.

3. **Agency-Specific Patterns:** Significant variations exist in response times across EMS, FDNY, and NYPD for similar incident types, suggesting opportunities for cross-agency coordination improvements.

4. **Incident Type Impact:** Life-threatening emergencies (e.g., cardiac arrests, major fires) show different response patterns compared to non-critical incidents, validating current prioritization systems but revealing potential for optimization.

5. **Temporal Trends:** Response times vary significantly by time of day, day of week, and season, indicating the need for dynamic resource allocation strategies that adapt to predictable demand patterns.

6. **Dispatch Efficiency:** Call-to-dispatch time and travel time emerged as critical predictors of overall response time across all regression models, highlighting these as key leverage points for operational improvements.

7. **Non-Linear Model Exploration:** Appendix analysis with Decision Trees and Random Forests (achieving 99.97% classification accuracy) demonstrated that while non-linear methods can capture complex patterns, the primary regression approaches provided more interpretable and actionable insights for operational optimization.

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

## Acknowledgments

- **Data Source:** City of New York Mayor's Office of Operations
- **Platform:** Analysis conducted using Deepnote collaborative environment
- **Dataset:** Available at [Data.gov - 911 End-to-End Data](https://catalog.data.gov/dataset/911-end-to-end-data)
