"""
Univariate Analysis Script for Heart Disease Prediction Project

Purpose:
- Create histograms for key features (Age, Cholesterol, Blood Pressure)
- Create pie chart showing class distribution of heart disease
- Save visualizations to 'results/' folder

Author: [Your Name]
Date: [Date]
"""

# =============================================
# 1. IMPORT REQUIRED LIBRARIES
# =============================================
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For creating static visualizations
import seaborn as sns  # For enhanced visualizations and styling
import os  # For operating system interactions (file/folder management)

# =============================================
# 2. CONFIGURATION SETTINGS
# =============================================
# Create results directory if it doesn't exist
# os.makedirs ensures we have a place to save our graphs
os.makedirs("results", exist_ok=True)

# Set visual style parameters for all plots
sns.set_theme(
    style="whitegrid",  # Use white background with grid lines
    palette="pastel",   # Use soft colors for better readability
    font_scale=1.1      # Slightly larger text for better visibility
)

# =============================================
# 3. LOAD AND INSPECT DATA
# =============================================
try:
    # Load preprocessed data from CSV file
    # We're using relative path: data/processed/standardized_data.csv
    heart_data = pd.read_csv("/Users/prajithravisankar/Documents/lakehead/bigData/project/heart-disease-prediction/data/processed/standardized_data.csv")
    
    # Quick check: Show first 3 rows to verify proper loading
    print("\nFirst 3 rows of dataset:")
    print(heart_data.head(3))
    
except FileNotFoundError as load_error:
    print("ERROR: Could not find data file!")
    print(f"Details: {load_error}")
    exit()

# =============================================
# 4. CREATE HISTOGRAMS FOR NUMERICAL FEATURES
# =============================================
# We'll create three histograms side-by-side using subplots

# Initialize figure with 1 row and 3 columns of plots
# figsize=(width, height) in inches - adjust for readability
histogram_figure, axis_array = plt.subplots(
    nrows=1, 
    ncols=3, 
    figsize=(20, 6)  # Wider figure to accommodate three plots
)

# --------------------------
# 4.1 Age Distribution Plot
# --------------------------
age_plot = sns.histplot(
    data=heart_data,
    x="Age",          # Use Age column for x-axis
    kde=True,         # Add Kernel Density Estimate line
    ax=axis_array[0], # Place in first position (leftmost)
    color="skyblue",  # Custom color for visual clarity
    edgecolor="black" # Add borders to bars
)

# Customize plot labels and titles
age_plot.set_title(
    "Standardized Age Distribution\n(Z-Score Normalized)", 
    pad=20,           # Add padding above title
    fontweight="bold"
)
age_plot.set_xlabel("Age (Standardized Z-Scores)", labelpad=15)
age_plot.set_ylabel("Number of Patients", labelpad=15)

# --------------------------
# 4.2 Cholesterol Distribution Plot
# --------------------------
chol_plot = sns.histplot(
    data=heart_data,
    x="Cholesterol",
    kde=True,
    ax=axis_array[1],  # Middle position
    color="salmon",
    edgecolor="black"
)

chol_plot.set_title(
    "Standardized Cholesterol Distribution\n(Z-Score Normalized)", 
    pad=20,
    fontweight="bold"
)
chol_plot.set_xlabel("Cholesterol (Standardized Z-Scores)", labelpad=15)
chol_plot.set_ylabel("")  # Remove redundant y-axis label

# --------------------------
# 4.3 Blood Pressure Distribution Plot
# --------------------------
bp_plot = sns.histplot(
    data=heart_data,
    x="Blood Pressure",
    kde=True,
    ax=axis_array[2],  # Rightmost position
    color="lightgreen",
    edgecolor="black"
)

bp_plot.set_title(
    "Standardized Blood Pressure Distribution\n(Z-Score Normalized)", 
    pad=20,
    fontweight="bold"
)
bp_plot.set_xlabel("Blood Pressure (Standardized Z-Scores)", labelpad=15)
bp_plot.set_ylabel("")  # Remove redundant y-axis label

# Adjust spacing between subplots
plt.tight_layout(pad=3.0)

# Save combined histograms to file
histogram_figure.savefig(
    "results/feature_distributions.png",
    dpi=300,           # High resolution for publications
    bbox_inches="tight" # Prevent cropping
)

# Clear figure from memory to save resources
plt.close(histogram_figure)

# =============================================
# 5. CREATE PIE CHART FOR CLASS DISTRIBUTION
# =============================================
# Count occurrences of each class (0 = No Disease, 1 = Disease)
class_counts = heart_data["Heart Disease"].value_counts()

# Create figure specifically for pie chart
plt.figure(figsize=(8, 8))  # Square figure for proper pie chart display

# Custom colors for clarity
colors = ["#66b3ff", "#ff9999"]  # Blue for healthy, red for disease
explode = (0.1, 0)               # "Explode" the diseased slice for emphasis

# Create pie chart with percentage labels
patches, text_labels, percentage_labels = plt.pie(
    class_counts,
    labels=["No Heart Disease", "Heart Disease"],
    colors=colors,
    autopct="%1.1f%%",    # Show percentages with 1 decimal
    startangle=90,        # Start first slice at top (12 o'clock)
    explode=explode,      # Separate the diseased slice
    shadow=True,          # Add depth with shadow
    textprops={"fontsize": 12}  # Larger percentage text
)

# Equal aspect ratio ensures pie is drawn as circle
plt.axis("equal")  

# Add descriptive title
plt.title(
    "Heart Disease Class Distribution in Dataset",
    pad=25,              # Extra padding below title
    fontsize=14,
    fontweight="bold"
)

# Save pie chart to file
plt.savefig(
    "results/class_balance.png",
    dpi=300,
    bbox_inches="tight"
)

# Clear pie chart from memory
plt.close()


"""
Expanded Univariate Analysis Script for Heart Disease Prediction

Purpose:
- Analyze distributions of all relevant features
- Identify potential patterns in both numerical and categorical variables
- Save visualizations to 'results/' folder

Author: [Your Name]
Date: [Date]
"""

# =============================================
#  CREATE HISTOGRAMS FOR ALL NUMERICAL FEATURES
# =============================================
# List of all numerical features from your dataset
numerical_features = [
    "Age", "Cholesterol", "Blood Pressure", 
    "Heart Rate", "Exercise Hours", "Stress Level",
    "Blood Sugar"
]

# Create figure with 2 rows and 4 columns for numerical features
plt.figure(figsize=(24, 12))
plt.suptitle("Distribution of Numerical Features", y=1.02, fontsize=16, fontweight="bold")

for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 4, i)
    sns.histplot(
        data=heart_data,
        x=feature,
        kde=True,
        color="teal",
        edgecolor="black"
    )
    plt.title(f"{feature} Distribution\n(Z-Score Normalized)", fontsize=12)
    plt.xlabel("Standardized Value", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/numerical_features_distributions.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================
# ANALYZE CATEGORICAL/DUMMY VARIABLES
# =============================================
# List of categorical/dummy variables from your dataset
categorical_features = [
    "Gender_Male", "Smoking_Former", "Smoking_Never",
    "Alcohol Intake_Moderate", "Family History_Yes",
    "Diabetes_Yes", "Obesity_Yes", "Exercise Induced Angina_Yes",
    "Chest Pain Type_Atypical Angina", 
    "Chest Pain Type_Non-anginal Pain",
    "Chest Pain Type_Typical Angina"
]

# Create bar plots for categorical features
plt.figure(figsize=(18, 20))
plt.suptitle("Categorical Feature Distributions", y=1.02, fontsize=16, fontweight="bold")

for i, feature in enumerate(categorical_features, 1):
    plt.subplot(4, 3, i)
    
    # Calculate percentages instead of counts
    percentages = heart_data[feature].value_counts(normalize=True) * 100
    
    # Create bar plot
    ax = sns.barplot(
        x=percentages.index,
        y=percentages.values,
        palette=["#4c72b0", "#55a868"],  # Custom colors for 0/1
        edgecolor="black"
    )
    
    # Add percentage labels
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width()/2.,
            height + 1,
            f'{height:.1f}%',
            ha="center",
            fontsize=10
        )
    
    # Customize plot
    plt.title(f"{feature} Distribution", fontsize=12)
    plt.xlabel("")
    plt.ylabel("Percentage", fontsize=10)
    plt.ylim(0, 100)  # Ensure consistent scale for comparison
    plt.xticks([0, 1], ["No", "Yes"] if feature != "Gender_Male" else ["Female", "Male"])
    plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/categorical_features_distributions.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================
# ANALYZE TARGET VARIABLE (HEART DISEASE)
# =============================================
# Enhanced target analysis with additional visualizations

# Create figure with 2 subplots
plt.figure(figsize=(16, 6))
plt.suptitle("Heart Disease Distribution Analysis", y=1.02, fontsize=16, fontweight="bold")

# Subplot 1: Pie chart (existing)
plt.subplot(1, 2, 1)
patches, texts, autotexts = plt.pie(
    class_counts,
    labels=["No Heart Disease", "Heart Disease"],
    colors=["#4c72b0", "#c44e52"],
    autopct="%1.1f%%",
    startangle=90,
    explode=(0.1, 0),
    shadow=True,
    textprops={"fontsize": 12}
)
plt.title("Class Balance", fontsize=14)
plt.axis("equal")

# Subplot 2: Boxplot comparison of key features vs target
plt.subplot(1, 2, 2)
sns.boxplot(
    x="Heart Disease",
    y="Blood Pressure",
    data=heart_data,
    palette=["#4c72b0", "#c44e52"],
    showfliers=False
)
plt.title("Blood Pressure Distribution by Heart Disease Status", fontsize=14)
plt.xlabel("Heart Disease", fontsize=12)
plt.ylabel("Blood Pressure (Standardized)", fontsize=12)
plt.xticks([0, 1], ["No Disease", "Disease"])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/target_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================
# ADDITIONAL UNIVARIATE ANALYSES
# =============================================
# Analysis 1: Stress Level Distribution by Gender
plt.figure(figsize=(10, 6))
sns.violinplot(
    x="Gender_Male",
    y="Stress Level",
    data=heart_data,
    palette=["#55a868", "#4c72b0"],  # Female=0, Male=1
    inner="quartile"
)
plt.title("Stress Level Distribution by Gender", fontsize=14)
plt.xlabel("Gender", fontsize=12)
plt.ylabel("Stress Level (Standardized)", fontsize=12)
plt.xticks([0, 1], ["Female", "Male"])
plt.grid(True, alpha=0.3)
plt.savefig("results/stress_by_gender.png", dpi=300, bbox_inches="tight")
plt.close()

# Analysis 2: Cholesterol vs Diabetes Status
plt.figure(figsize=(10, 6))
sns.boxenplot(
    x="Diabetes_Yes",
    y="Cholesterol",
    data=heart_data,
    palette=["#4c72b0", "#c44e52"],  # No=0, Yes=1
    showfliers=False
)
plt.title("Cholesterol Distribution by Diabetes Status", fontsize=14)
plt.xlabel("Diabetes", fontsize=12)
plt.ylabel("Cholesterol (Standardized)", fontsize=12)
plt.xticks([0, 1], ["No Diabetes", "Diabetes"])
plt.grid(True, alpha=0.3)
plt.savefig("results/cholesterol_diabetes.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================
# FINAL CONFIRMATION
# =============================================
print("\n" + "="*50)
print("ANALYSIS COMPLETE".center(50))
print("="*50)
print("\nGenerated files:")
print(f"- Feature distributions: results/feature_distributions.png")
print(f"- Class balance: results/class_balance.png")