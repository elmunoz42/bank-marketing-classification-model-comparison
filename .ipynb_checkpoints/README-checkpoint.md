# Bank Marketing Classification Model Comparison

## Overview
This project compares the performance of various classification models (K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines) on a banking telemarketing dataset. The goal is to predict whether a client will subscribe to a term deposit based on various client, campaign, and socio-economic attributes.

## Dataset
The dataset comes from the UCI Machine Learning repository and contains information about a Portuguese banking institution's marketing campaigns. It includes:
- Client demographic information (age, job, marital status, education)
- Financial attributes (default status, housing loan, personal loan)
- Current campaign contact details (contact type, month, day, duration)
- Previous campaign information (previous contacts, outcomes)
- Social and economic context attributes (employment variation rate, consumer indices)

The target variable is whether the client subscribed to a term deposit ('yes' or 'no').

## Methodology
The project follows these key steps:

1. **Data Understanding** - Exploring and understanding the data structure, distributions, and potential relationships between features and the target variable.

2. **Data Preparation**
   - Handling missing and unknown values
   - Feature engineering (creating a touch_point feature from pdays)
   - Encoding categorical variables (one-hot encoding for most categorical features)
   - Scaling numerical features

3. **Modeling**
   - Building a pipeline for preprocessing and model training
   - Training and evaluating four different classification models:
     - K-Nearest Neighbors
     - Logistic Regression
     - Decision Trees
     - Support Vector Machines
   - Using cross-validation to ensure model reliability

4. **Evaluation**
   - Comparing models based on accuracy, precision, recall, F1-score, and AUC-ROC
   - Analyzing feature importance to understand key drivers of term deposit subscriptions
   - Creating visualizations to illustrate model performance differences

## Key Findings
- [To be filled with your project's actual findings]
- Comparison of model accuracies, advantages, and limitations
- Most important features for predicting term deposit subscriptions
- Recommendations for optimizing future marketing campaigns

## Technical Implementation
- Python 3.x with standard data science libraries (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Jupyter Notebook for interactive analysis and visualization
- Pipeline architecture for reproducible preprocessing and model training
- Feature engineering and selection techniques to improve model performance

## File Structure
- `prompt_III.ipynb`: Main Jupyter notebook containing the analysis
- `README.md`: Project documentation
- `data/`: Directory containing the dataset
- `images/`: Directory containing generated plots and figures
- `function.py`: Auxiliary Python functions to aid development
- `claude-statistical-analysis.txt`: Output from Claude API statistical analysis experiment

## Future Work
- Experiment with more advanced models (e.g., ensemble methods, neural networks)
- Perform more extensive hyperparameter tuning
- Explore additional feature engineering opportunities
- Develop a simple web application for real-time predictions

## References
1. Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31.
2. UCI Machine Learning Repository: [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing)