import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from itertools import combinations
from joblib import Parallel, delayed
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from itertools import combinations
from joblib import Parallel, delayed
import os
import multiprocessing
from matplotlib.backends.backend_pdf import PdfPages
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import re
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# -------- Data loading --------
file_path = r"C:\Users\27955\Desktop\Dissertation\dataset\3\V1datasetAttrition.csv"
data = pd.read_csv(file_path)

# -------- Data preprocessing --------
data['BusinessTravel'] = data['BusinessTravel'].replace({
    'Travel_Rarely': 'Rare',
    'Travel_Frequently': 'Freq',
    'Non-Travel': 'None'
})
data['Department'] = data['Department'].replace({
    'Research & Development': 'R&D',
    'Human Resources': 'HR',
    'Sales': 'Sales'
})
data['JobRole'] = data['JobRole'].replace({
    'Sales Representative': 'SalesRep',
    'Research Scientist': 'ResScientist',
    'Laboratory Technician': 'LabTech',
    'Healthcare Representative': 'HealthRep',
    'Manufacturing Director': 'ManufDir',
    'Manager': 'Manager',
    'Research Director': 'ResDir',
    'Human Resources': 'HR'
})
data['Source_of_Hire'] = data['Source_of_Hire'].replace({
    'Job Event': 'Event',
    'Job Portal': 'Portal',
    'Recruiter': 'Recruit',
    'Walk-in': 'WalkIn'
})

categorical_columns = [
    'BusinessTravel', 'Department', 'Gender', 'MaritalStatus',
    'OverTime', 'JobRole', 'Mode_of_work', 'Source_of_Hire', 'Job_mode',
    'Work_accident', 'Higher_Education(12th=1,Graduation=2,Post-Graduation=3,PHD=4)',
    'Status_of_leaving', 'StockOptionLevel(0-3)'
]
for col in categorical_columns:
    data[col] = data[col].astype('category')

# -------- Add interaction terms --------
# Encode categorical variables first
data['Higher_Education'] = data['Higher_Education(12th=1,Graduation=2,Post-Graduation=3,PHD=4)'].cat.codes
data['OverTime_Code'] = data['OverTime'].cat.codes

# Construct interaction terms
data['Age_JobSatisfaction'] = data['Age'] * data['JobSatisfaction(1-4)']
data['OverTime_JobSatisfaction'] = data['OverTime_Code'] * data['JobSatisfaction(1-4)']
data['Education_JobLevel'] = data['Higher_Education'] * data['JobLevel(1-5)']

# -------- Delete redundant features --------
# Remove redundant features that have been replaced by interaction terms
features_to_drop = ['Age', 'JobSatisfaction(1-4)', 'JobLevel(1-5)']
data = data.drop(columns=features_to_drop)

# -------- Data visualization part --------
sns.set_theme(style="whitegrid")
main_color = '#1f77b4'

# Numeric variables - split into two 4x2 plots
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns[:16]  # 前16个变量
chunks = [numerical_columns[:8], numerical_columns[8:16]]
filenames = ['numeric-vis-1.png', 'numeric-vis-2.png']

# for idx, cols in enumerate(chunks):
#     plt.figure(figsize=(14, 16))  # 每图 4x2
#     for i, col in enumerate(cols, 1):
#         plt.subplot(4, 2, i)
#         sns.histplot(data[col], kde=True, bins=30, color=main_color, edgecolor='black')
#         plt.title(f'Distribution of {col}', fontsize=18)
#         plt.xlabel(col, fontsize=15)
#         plt.ylabel('Frequency', fontsize=15)
#     plt.tight_layout()
#     plt.savefig(filenames[idx], dpi=300)
#     plt.close()
#
#
# # Categorical variable distribution visualization
# for i in range(0, len(categorical_columns), 6):
#     plt.figure(figsize=(18, 12))
#     for j, col in enumerate(categorical_columns[i:i+6], 1):
#         plt.subplot(2, 3, j)
#         ax = sns.countplot(x=data[col], edgecolor='black', color=main_color)
#         plt.title(f'Count of {col}', fontsize=18)
#         plt.xlabel(col, fontsize=15)
#         plt.ylabel('Count', fontsize=15)
#         plt.xticks(rotation=30, fontsize=14)
#         for p in ax.patches:
#             ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
#                         ha='center', va='baseline', fontsize=9, color='black', xytext=(0, 5), textcoords='offset points')
    # plt.tight_layout()
    # plt.show()


# # Distribution visualization of the target variable Attrition
# plt.figure(figsize=(8, 6))
# ax = sns.countplot(x=data['Attrition'], edgecolor='black', color=main_color)
# plt.title('Attrition Distribution (0: Stayed, 1: Left)', fontsize=16)
# plt.xlabel('Attrition', fontsize=12)
# plt.ylabel('Count', fontsize=12)
# plt.xticks(ticks=[0, 1], labels=['Stayed', 'Left'], fontsize=10)
#
# for p in ax.patches:
#     ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
# plt.tight_layout()
# plt.show()

# -------- VIF analysis --------
# Separate features and labels
X = data.drop(columns=['Attrition'])
y = data['Attrition']


def calculate_vif(X):
    X_numeric = X.select_dtypes(include=['float64', 'int64'])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_numeric.columns
    vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
    return vif_data


vif_result = calculate_vif(X)
print("VIF without One-Hot Encoding:")
print(vif_result)

# -------- Dataset division --------
# Split training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------- Undersampling --------
# Divide the training set into majority class and minority class
data_train = pd.concat([X_train, y_train], axis=1)
data_majority = data_train[data_train['Attrition'] == 0]
data_minority = data_train[data_train['Attrition'] == 1]

# Undersample the majority class
data_majority_downsampled = resample(data_majority,
                                     replace=False,
                                     n_samples=len(data_minority),
                                     random_state=42)

#Merge the majority and minority classes after undersampling
data_balanced = pd.concat([data_majority_downsampled, data_minority])

# Separate balanced features and labels
X_train_balanced = data_balanced.drop(columns=['Attrition'])
y_train_balanced = data_balanced['Attrition']

# -------- Lifetime and event indicator construction --------
# Use YearsAtCompany as the lifetime and Attrition as the event indicator
data['Survival_Time'] = data['YearsAtCompany']  # 生存时间
data['Event_Indicator'] = data['Attrition']  # 事件指示符（离职 = 1，未离职 = 0）

# -------- KM --------
time_col = 'Survival_Time'
event_col = 'Event_Indicator'

alpha = 0.05
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
exclude_vars = ['Date_of_Hire', 'Status_of_leaving']
categorical_cols = [col for col in categorical_cols if col not in exclude_vars]

# Automatically obtain the number of CPU - 1
cpu_count = max(multiprocessing.cpu_count() - 1, 1)
print(f"Using {cpu_count} CPU cores for parallel processing.")

# Define a function to perform a log-rank test on a category combination and return the result
def compare_two_groups(data, cat_var, cat1, cat2, time_col, event_col):
    group1 = data[data[cat_var] == cat1]
    group2 = data[data[cat_var] == cat2]

    # If there is insufficient data, return None
    if len(group1) == 0 or len(group2) == 0:
        return None

    # Perform Log-Rank test
    result = logrank_test(
        durations_A=group1[time_col],
        durations_B=group2[time_col],
        event_observed_A=group1[event_col],
        event_observed_B=group2[event_col]
    )
    p_val = result.p_value
    return (cat_var, f"{cat1} vs {cat2}", p_val)

# Prepare the task list
tasks = []
for cat_var in categorical_cols:
    categories = data[cat_var].dropna().unique()
    if len(categories) < 2:
        continue
    for cat1, cat2 in combinations(categories, 2):
        tasks.append((cat_var, cat1, cat2))

# Process all comparison tasks in parallel
results = Parallel(n_jobs=cpu_count, verbose=10)(
    delayed(compare_two_groups)(data, cat_var, cat1, cat2, time_col, event_col)
    for cat_var, cat1, cat2 in tasks
)

# Arrange the results
results_summary = {}
for res in results:
    if res is None:
        continue
    var, combo, p_val = res
    if var not in results_summary:
        results_summary[var] = []
    results_summary[var].append((combo, p_val))

significant_vars = [var for var, comps in results_summary.items() if any(p < alpha for _, p in comps)]

# Output analysis results
print("===== KM analysis (Log-Rank test) results =====")
for var, comparisons in results_summary.items():
    if not comparisons:
        print(f"{var}: No valid comparison between groups")
        continue
    print(f"\nvariable: {var}")
    for combo, p_val in comparisons:
        print(f"  {combo}: p-value={p_val:.4e}")

print("\n===== List of significant variables (p-value < 0.05 in any category comparison) =====")
if significant_vars:
    for v in significant_vars:
        print(f"- {v}")
else:
    print("No significant variables")

def plot_km_one_by_one(data, features, time_col, event_col):
# Separate Kaplan-Meier survival curves were plotted for each significant variable.
    for feature in features:
        plt.figure(figsize=(8, 6))
        kmf = KaplanMeierFitter()

        for category in data[feature].dropna().unique():
            mask = data[feature] == category
            if mask.sum() == 0:
                continue
            kmf.fit(
                durations=data[time_col][mask],
                event_observed=data[event_col][mask],
                label=str(category)
            )
            kmf.plot(ci_show=False)

        plt.title(f"KM Curve: {feature}", fontsize=12)
        plt.xlabel("Time (Years)", fontsize=10)
        plt.ylabel("Survival Probability", fontsize=10)
        plt.legend(title=feature, fontsize=8, title_fontsize=9, loc="lower left")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

# # ---------- Calling drawing functions ----------
# plot_km_one_by_one(data, significant_vars, time_col, event_col)

# Visualization
print("Survival Time Distribution:")
print(data['Survival_Time'].describe())

# Check the distribution of event indicators
print("Event Indicator Distribution:")
print(data['Event_Indicator'].value_counts())

# # Visualizing the survival time distribution
# plt.figure(figsize=(8, 6))
# sns.histplot(data['Survival_Time'], bins=20, kde=True, color='blue')
# plt.title('Distribution of Survival Time (YearsAtCompany)', fontsize=14)
# plt.xlabel('Years at Company', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.tight_layout()
# plt.show()
#
# # Visualize the impact of different features on survival time
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Department', y='Survival_Time', hue='Attrition', data=data)
# plt.title('Survival Time by Department and Attrition Status', fontsize=14)
# plt.xlabel('Department', fontsize=12)
# plt.ylabel('Years at Company', fontsize=12)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# -------- Cox（V2） --------
# One-hot encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)


# Step 1: Define functions to calculate VIF and check proportional hazards assumption
def calculate_vif(features):
# Calculates the variance inflation factor (VIF) for numerical features.
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features.columns
    vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    return vif_data


def drop_high_vif_features(features, threshold=10):
    vif_data = calculate_vif(features)
    high_vif_features = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()
    print(f"Remove features whose VIF exceeds {threshold}: {high_vif_features}")
    return features.drop(columns=high_vif_features, errors='ignore')

# Step 2: Define features and target variables for Cox regression
cox_features = data_encoded.drop(columns=['Survival_Time', 'Event_Indicator', 'YearsAtCompany', 'Attrition'],
                                 errors='ignore')
cox_target = data_encoded[['Survival_Time', 'Event_Indicator']]

# Step 3: Remove features with multicollinearity issues
numerical_features = cox_features.select_dtypes(include=['float64', 'int64'])
print("Initial VIF calculation results:")
print(calculate_vif(numerical_features))
cox_features = drop_high_vif_features(numerical_features, threshold=10)

# Step 4: Add stratified variables and time interaction terms
# For variables that do not meet the PH assumption, add stratified and time interaction terms
cox_features['YearsSinceLastPromotion_bins'] = pd.cut(data['YearsSinceLastPromotion'], bins=3, labels=False)
cox_features['YearsSinceLastPromotion_time'] = cox_target['Survival_Time'] * cox_features['YearsSinceLastPromotion']

# Add other interaction terms, keeping only columns that exist in the print result

# 1. Add interaction terms only if 'Department_R&D' and 'Department_Sales' exist
if 'Department_R&D' in data_encoded.columns:
    cox_features['Department_R&D_time'] = data_encoded['Department_R&D'] * data_encoded['Survival_Time']

if 'Department_Sales' in data_encoded.columns:
    cox_features['Department_Sales_time'] = data_encoded['Department_Sales'] * data_encoded['Survival_Time']

# 2. Add interaction term only if 'OverTime_Yes' and 'JobSatisfaction(1-4)' exist
if 'OverTime_Yes' in data_encoded.columns and 'JobSatisfaction(1-4)' in data_encoded.columns:
    cox_features['OverTime_JobSatisfaction'] = data_encoded['OverTime_Yes'] * data_encoded['JobSatisfaction(1-4)']

# 3. Add interaction term only if 'Higher_Education' and 'JobLevel(1-5)' exist
if 'Higher_Education' in data_encoded.columns and 'JobLevel(1-5)' in data_encoded.columns:
    cox_features['Education_JobLevel'] = data_encoded['Higher_Education'] * data_encoded['JobLevel(1-5)']

# Step 4.5: Check the VIF again after adding the interaction term
updated_numerical = cox_features.select_dtypes(include=['float64', 'int64'])
print("The VIF recalculation results after the introduction of the interaction term are:")
print(calculate_vif(updated_numerical))

# Step 5: Fit the Cox model and check the proportional hazards assumption
cox_model = CoxPHFitter()

try:
    print("Begin fitting Cox model (with stratification)...")
    cox_model.fit(
        pd.concat([cox_target, cox_features], axis=1),
        duration_col='Survival_Time',
        event_col='Event_Indicator',
        strata=['YearsSinceLastPromotion_bins']
    )
    print("Initial Cox model results (with stratification):")
    cox_model.print_summary()

    # Checking the proportional hazards assumption
    print("Checking the proportional hazards assumption...")
    cox_model.check_assumptions(
        pd.concat([cox_target, cox_features], axis=1),
        p_value_threshold=0.05,
        show_plots=False
    )
except Exception as e:
    print(f"Initial fit model failed: {e}")

# Step 6: Model evaluation: Calculate C-index
print(f"C-index: {cox_model.concordance_index_:.2f}")

# Step 7: Output the final model results
print("Finally, the Cox model is completed and all steps are executed.")

# RSF model (long term, predicting when to churn)
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance

# Process the data to ensure that X_survival contains only numeric data
categorical_cols = data.select_dtypes(include=['category', 'object']).columns.tolist()
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Selecting variables related to survival analysis
survival_data = data_encoded[['Survival_Time', 'Event_Indicator']].copy()
X_survival = data_encoded.drop(columns=['Survival_Time', 'Event_Indicator', 'Attrition'])

# Make sure Event_Indicator is a Boolean type
survival_data["Event_Indicator"] = survival_data["Event_Indicator"].astype(bool)

# Training/testing set split
X_train_rsf, X_test_rsf, y_train_rsf, y_test_rsf = train_test_split(
    X_survival, survival_data, test_size=0.3, random_state=42
)

# Convert y_train_rsf and y_test_rsf to structured arrays
y_train_rsf = np.array(
    [(event, time) for event, time in zip(y_train_rsf["Event_Indicator"], y_train_rsf["Survival_Time"])],
    dtype=[('event', '?'), ('time', '<f8')]
)

y_test_rsf = np.array(
    [(event, time) for event, time in zip(y_test_rsf["Event_Indicator"], y_test_rsf["Survival_Time"])],
    dtype=[('event', '?'), ('time', '<f8')]
)

# Training a Random Survival Forest Model
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, random_state=42)
rsf.fit(X_train_rsf, y_train_rsf)

# Calculate C-index to evaluate the model
c_index_rsf = rsf.score(X_test_rsf, y_test_rsf)
print(f"Random Survival Forest (RSF) C-index: {c_index_rsf:.3f}")

# Calculate feature importance (using permutation importance)
result = permutation_importance(
    rsf, X_test_rsf, y_test_rsf, n_repeats=10, random_state=42, n_jobs=-1
)

# Get feature importance ranking
sorted_idx = result.importances_mean.argsort()[::-1]

# Print the top 10 most important features
print("\nRSF feature importance (top 10):")
for i in sorted_idx[:10]:
    print(f"{X_train_rsf.columns[i]}: {result.importances_mean[i]:.4f}")

from sksurv.metrics import integrated_brier_score
import numpy as np

# Calculate survival probability prediction
survival_predictions = rsf.predict_survival_function(X_test_rsf)

# Convert StepFunction to 2D array
time_points = np.percentile(y_test_rsf["time"], np.linspace(5, 95, 10))
survival_prob_matrix = np.array([[fn(t) for t in time_points] for fn in survival_predictions])

# Calculating the Brier Score
brier_score = integrated_brier_score(
    y_train_rsf, y_test_rsf, survival_prob_matrix, time_points
)

print(f"\nRSF Integrated Brier Score (IBS): {brier_score:.4f}")

from sksurv.metrics import cumulative_dynamic_auc
import matplotlib.pyplot as plt

# Select multiple time points (from 5% to 95% of survival time)
time_points = np.percentile(y_test_rsf["time"], np.linspace(5, 95, 10))

# Calculate AUC
auc_scores, auc_errors = cumulative_dynamic_auc(
    y_train_rsf, y_test_rsf, rsf.predict(X_test_rsf), time_points
)

# # Plotting time-dependent AUC curves
# plt.figure(figsize=(8, 6))
# plt.plot(time_points, auc_scores, marker="o", linestyle="-", label="RSF AUC")
# plt.fill_between(time_points, auc_scores - auc_errors, auc_scores + auc_errors, alpha=0.2)
# plt.axhline(y=0.9, color="red", linestyle="--", linewidth=1,label="AUC = 0.90")
# plt.xlabel("Time")
# plt.ylabel("AUC")
# plt.title("Time-dependent AUC for RSF")
# plt.legend()
# plt.show()

print(f"RSF Time-dependent AUC at different time points: {auc_scores}")

# ======================================================
times_rsf   = getattr(rsf, "event_times_", rsf.unique_times_)
surv_fns    = rsf.predict_survival_function(X_test_rsf)

p3_list, p5_list = [], []

for fn in surv_fns:
    S3 = fn(3)
    S5 = fn(5)
    p3_list.append(1 - S3)
    p5_list.append(1 - S5)

prob_df = pd.DataFrame({
    "Emp_ID" : X_test_rsf.index,
    "Prob_≤3yr": np.round(p3_list, 3),
    "Prob_≤5yr": np.round(p5_list, 3)
})
prob_df.to_csv("rsf_emp_window_probs.csv", index=False)
print(prob_df.head())

# XGBoost model (elimination)
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Prepare data (using Attrition as a binary target variable)
X_classification = X_survival
y_classification = data_encoded["Attrition"]  # 0 = not lost, 1 = lost

# Divide into training set and test set
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_classification, y_classification, test_size=0.3, random_state=42
)

# Oversampling churned employees (using SMOTE)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_xgb, y_train_xgb)

# Calculate class weights (scale_pos_weight)
class_0_count = sum(y_train_smote == 0)
class_1_count = sum(y_train_smote == 1)
scale_pos_weight = class_0_count / class_1_count  # Let the model pay more attention to category 1

# Train the XGBoost model
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

xgb.fit(X_train_smote, y_train_smote)

# Prediction results
y_pred_xgb = xgb.predict(X_test_xgb)
y_pred_proba_xgb = xgb.predict_proba(X_test_xgb)[:, 1]

# Calculate evaluation metrics
auc_xgb = roc_auc_score(y_test_xgb, y_pred_proba_xgb)
f1_xgb = f1_score(y_test_xgb, y_pred_xgb)
accuracy_xgb = accuracy_score(y_test_xgb, y_pred_xgb)

print(f"\nXGBoost：")
print(f"AUC: {auc_xgb:.3f}")
print(f"F1-score: {f1_xgb:.3f}")
print(f"Accuracy: {accuracy_xgb:.3f}")

# Print Classification Report
print("\nClassification Report：")
print(classification_report(y_test_xgb, y_pred_xgb))

# # Plotting the confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(confusion_matrix(y_test_xgb, y_pred_xgb), annot=True, fmt="d", cmap="Blues",
#             xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("XGBoost Confusion Matrix")
# plt.show()

# LightGBM model (short-term, slightly inferior to catboost)
import re
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Read data
file_path = r"C:\Users\27955\Desktop\Dissertation\dataset\3\V1datasetAttrition.csv"
data = pd.read_csv(file_path)

# Select categorical and numerical variables
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove("Attrition")  # Target variable is not used as a feature

# One-Hot
data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

X = data_encoded.drop(columns=['Attrition'])
y = data_encoded['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lgb_classifier = LGBMClassifier(
    boosting_type="gbdt",
    objective="binary",
    metric="auc",
    learning_rate=0.01,
    num_leaves=70,
    max_depth=6,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=10,
    scale_pos_weight=2.1,
    n_estimators=500,
    random_state=42
)

X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]
X_test.columns = [re.sub(r'[^\w]', '_', col) for col in X_test.columns]

# Train the model
lgb_classifier.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[lgb.log_evaluation(10), lgb.early_stopping(100)]
)

best_iteration = lgb_classifier.best_iteration_
print(f"Optimal number of training rounds: {best_iteration}")

# Predicted probability
y_pred_proba = lgb_classifier.predict_proba(X_test)[:, 1]

# Use PR curve to find the best classification threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[f1_scores.argmax()]
print(f"Optimal classification threshold: {best_threshold:.3f}")

# Reuse the best threshold
y_pred = (y_pred_proba > best_threshold).astype(int)

# Calculating evaluation metrics
auc_lgb = roc_auc_score(y_test, y_pred_proba)
f1_lgb = f1_score(y_test, y_pred)
accuracy_lgb = accuracy_score(y_test, y_pred)

print(f"\nLightGBM：")
print(f"AUC: {auc_lgb:.3f}")
print(f"F1-score: {f1_lgb:.3f}")
print(f"Accuracy: {accuracy_lgb:.3f}")

# Print Classification Report
print("\nClassification Report：")
print(classification_report(y_test, y_pred, zero_division=1))

# # Plot the confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
#             xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("LightGBM Confusion Matrix")
# plt.show()

# CatBoost (short-term, direct classification of loss)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_curve

data = pd.read_csv(file_path)

categorical_features = data.select_dtypes(include=['object']).columns.tolist()
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove("Attrition")

# One-Hot
data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

X = data_encoded.drop(columns=['Attrition'])
y = data_encoded['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

catboost_model = CatBoostClassifier(
    iterations=3000,
    depth=6,
    learning_rate=0.01,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=200,
    class_weights=[1, 7]
)

catboost_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)

best_iteration = catboost_model.best_iteration_
print(f"Optimal number of training rounds: {best_iteration}")

y_pred_proba_catboost = catboost_model.predict_proba(X_test)[:, 1]

# Calculate the optimal classification threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_catboost)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[f1_scores.argmax()]
print(f"Optimal classification threshold: {best_threshold:.3f}")

# Apply the best threshold to make final predictions
y_pred_catboost = (y_pred_proba_catboost > best_threshold).astype(int)

# Calculate evaluation metrics
auc_catboost = roc_auc_score(y_test, y_pred_proba_catboost)
f1_catboost = f1_score(y_test, y_pred_catboost)
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)

print(f"\nCatBoost：")
print(f"AUC: {auc_catboost:.3f}")
print(f"F1-score: {f1_catboost:.3f}")
print(f"Accuracy: {accuracy_catboost:.3f}")

# Print Classification Report
print("\nClassification Report：")
print(classification_report(y_test, y_pred_catboost))

# # Plot the confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(confusion_matrix(y_test, y_pred_catboost), annot=True, fmt="d", cmap="Blues",
#             xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("CatBoost Confusion Matrix")
# plt.show()

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_curve

# ========== **Short-term prediction (whether to lose)** ==========

# Prediction probability stacking as a new feature
stacked_proba = np.vstack((y_pred_proba, y_pred_proba_catboost)).T

# Fitting the meta-learner (Stacking Fusion)
from sklearn.linear_model import LogisticRegression
stacking_meta_model = LogisticRegression()
stacking_meta_model.fit(stacked_proba, y_test)

# Fusion prediction
final_pred_proba = stacking_meta_model.predict_proba(stacked_proba)[:, 1]

# Find the best threshold
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, roc_auc_score
precision, recall, thresholds = precision_recall_curve(y_test, final_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[f1_scores.argmax()]

# Final classification
final_pred = (final_pred_proba > best_threshold).astype(int)

# Model evaluation
auc_final = roc_auc_score(y_test, final_pred_proba)
f1_final = f1_score(y_test, final_pred)
accuracy_final = accuracy_score(y_test, final_pred)

print(f"Fusion Model AUC: {auc_final:.3f}")
print(f"Fusion Model F1-score: {f1_final:.3f}")
print(f"Fusion Model Accuracy: {accuracy_final:.3f}")
print(f"Optimal classification threshold: {best_threshold:.3f}")

# Draw the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, final_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fusion Model Confusion Matrix (Short-term)")
plt.show()

shortterm_df = pd.DataFrame({
    "Emp_ID": X_test.index,   # 可改为员工编号列（如 data['EmployeeID']）如果有
    "Pred_Prob": np.round(final_pred_proba, 3),
    "Prediction": final_pred,
    "Risk_Level": ["High" if prob > best_threshold else "Low" for prob in final_pred_proba]
})

print("\nShort-term Prediction Results (Fusion Model):")
print(shortterm_df.head())

# ========== Long-term prediction (churn time) — RSF only ==========

# Use previously trained RSF to predict survival time
rsf_time_pred = rsf.predict(X_test_rsf)

# Calculate C-index to evaluate the accuracy of long-term forecasts
c_index = concordance_index_censored(y_test_rsf['event'], y_test_rsf['time'], rsf_time_pred)[0]
print(f"\nC-index of RSF predicting survival time: {c_index:.3f}")
