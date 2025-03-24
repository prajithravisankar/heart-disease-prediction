"""
Enhanced Univariate Analysis for Heart Disease Prediction
Using Raw Clinical Values from cleaned_data.csv
"""

# =============================================
# 1. IMPORT REQUIRED LIBRARIES
# =============================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================================
# 2. CONFIGURATION SETTINGS
# =============================================
os.makedirs("results", exist_ok=True)

sns.set_theme(
    style="whitegrid",
    palette="pastel",
    font_scale=1.1,
    rc={"axes.titlesize": 14, "axes.labelsize": 12}
)

# Clinical thresholds dictionary
CLINICAL_THRESHOLDS = {
    "Cholesterol": {
        "Borderline High (200 mg/dL)": 200,
        "High (240 mg/dL)": 240
    },
    "Blood Pressure": {
        "Hypertension Threshold (140 mmHg)": 140
    },
    "Blood Sugar": {
        "Normal Fasting (<100 mg/dL)": 100,
        "Prediabetes (100-125 mg/dL)": 125
    }
}

# =============================================
# 3. LOAD CLEANED DATA
# =============================================
try:
    heart_data = pd.read_csv("../data/processed/cleaned_data.csv")
    print("\nData sample:")
    print(heart_data[["Age", "Cholesterol", "Blood Pressure", "Heart Disease"]].head(3))
except FileNotFoundError as e:
    print(f"Data loading error: {e}")
    exit()

# =============================================
# 4. CLINICAL DISTRIBUTION ANALYSIS
# =============================================
def add_clinical_guidelines(ax, feature):
    """Add reference lines for clinical thresholds"""
    if feature in CLINICAL_THRESHOLDS:
        for label, value in CLINICAL_THRESHOLDS[feature].items():
            ax.axvline(value, color='darkred', linestyle='--', alpha=0.7)
            ax.text(
                x=value + (0.02 * ax.get_xlim()[1]),
                y=ax.get_ylim()[1]*0.9,
                s=label,
                rotation=90,
                verticalalignment='top',
                color='darkred'
            )

# Initialize figure
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# --------------------------
# 4.1 Age Distribution
# --------------------------
age_plot = sns.histplot(
    data=heart_data,
    x="Age",
    bins=15,
    kde=True,
    ax=axs[0],
    color="skyblue",
    edgecolor="black"
)
age_plot.set_title("Age Distribution of Patients", fontweight="bold")
age_plot.set_xlabel("Age (Years)", labelpad=15)
age_plot.set_ylabel("Count", labelpad=15)

# --------------------------
# 4.2 Cholesterol Analysis
# --------------------------
chol_plot = sns.histplot(
    data=heart_data,
    x="Cholesterol",
    bins=20,
    kde=True,
    ax=axs[1],
    color="salmon",
    edgecolor="black"
)
chol_plot.set_title("Cholesterol Distribution with Clinical Guidelines", fontweight="bold")
chol_plot.set_xlabel("Cholesterol (mg/dL)", labelpad=15)
add_clinical_guidelines(chol_plot, "Cholesterol")

# --------------------------
# 4.3 Blood Pressure Analysis
# --------------------------
bp_plot = sns.histplot(
    data=heart_data,
    x="Blood Pressure",
    bins=15,
    kde=True,
    ax=axs[2],
    color="lightgreen",
    edgecolor="black"
)
bp_plot.set_title("Blood Pressure Distribution with Hypertension Threshold", fontweight="bold")
bp_plot.set_xlabel("Systolic Blood Pressure (mmHg)", labelpad=15)
add_clinical_guidelines(bp_plot, "Blood Pressure")

plt.tight_layout()
plt.savefig("../results/clinical_distributions.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================
# 5. ENHANCED CLASS ANALYSIS
# =============================================
plt.figure(figsize=(12, 6))

# Subplot 1: Pie Chart
plt.subplot(1, 2, 1)
class_counts = heart_data["Heart Disease"].value_counts()
patches, texts, autotexts = plt.pie(
    class_counts,
    labels=["Healthy", "Heart Disease"],
    colors=["#4c72b0", "#c44e52"],
    autopct=lambda p: f"{p:.1f}%\n({int(p*sum(class_counts))/100})",
    startangle=90,
    explode=(0.1, 0),
    shadow=True,
    textprops={"fontsize": 10}
)
plt.title("Class Distribution", fontweight="bold")

# Subplot 2: Age vs Disease Status
plt.subplot(1, 2, 2)
sns.boxplot(
    x="Heart Disease",
    y="Age",
    hue="Heart Disease",  # Add hue parameter
    data=heart_data,
    palette=["#4c72b0", "#c44e52"],
    showfliers=False,
    width=0.5,
    legend=False  # Suppress legend
)
plt.title("Age Distribution by Heart Disease Status", fontweight="bold")
plt.xlabel("Heart Disease Diagnosis", labelpad=10)
plt.ylabel("Age (Years)", labelpad=10)
plt.xticks([0, 1], ["No Disease", "Disease Present"])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../results/enhanced_class_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================
# 6. COMPREHENSIVE RISK FACTOR ANALYSIS
# =============================================
plt.figure(figsize=(20, 12))  # Increased figure size
plt.suptitle("Key Risk Factor Distributions", y=1.02, fontweight="bold")

# Blood Sugar Analysis
plt.subplot(2, 2, 1)
sugar_plot = sns.histplot(
    data=heart_data,
    x="Blood Sugar",
    bins=15,
    kde=True,
    color="purple",
    edgecolor="black"
)
add_clinical_guidelines(sugar_plot, "Blood Sugar")
plt.title("Fasting Blood Sugar Distribution", fontweight="bold")
plt.xlabel("Blood Sugar (mg/dL)", fontsize=10)

# Stress Level Analysis
plt.subplot(2, 2, 2)
sns.boxplot(
    x="Heart Disease",
    y="Stress Level",
    hue="Heart Disease",
    data=heart_data,
    palette=["#4c72b0", "#c44e52"],
    showfliers=False,
    legend=False
)
plt.title("Stress Levels by Heart Disease Status", fontweight="bold")
plt.xlabel("Heart Disease Diagnosis", fontsize=10)
plt.ylabel("Stress Level (1-10 Scale)", fontsize=10)

# Cholesterol vs Blood Pressure
plt.subplot(2, 2, 3)
sns.scatterplot(
    data=heart_data,
    x="Cholesterol",
    y="Blood Pressure",
    hue="Heart Disease",
    palette=["#4c72b0", "#c44e52"],
    alpha=0.8,
    edgecolor="black"
)
plt.title("Cholesterol vs Blood Pressure Relationship", fontweight="bold")
plt.xlabel("Cholesterol (mg/dL)", fontsize=10)
plt.ylabel("Blood Pressure (mmHg)", fontsize=10)

# Risk Factor Prevalence
plt.subplot(2, 2, 4)
risk_factors = heart_data[["Smoking_Former", "Diabetes_Yes", "Obesity_Yes"]].mean() * 100
risk_factors.plot(kind="bar", color=["#ff9999", "#66b3ff", "#99ff99"], edgecolor="black")
plt.title("Prevalence of Major Risk Factors", fontweight="bold")
plt.xticks(
    ticks=[0, 1, 2],
    labels=["Smoking History", "Diabetes", "Obesity"],
    rotation=45
)
plt.ylabel("Percentage of Patients (%)", fontsize=10)
plt.ylim(0, 100)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
plt.savefig("../results/risk_factor_analysis.png", dpi=300, bbox_inches="tight")
plt.close()


print("\nAnalysis Complete. Generated Reports:")
print("- Clinical distributions: results/clinical_distributions.png")
print("- Class analysis: results/enhanced_class_analysis.png")
print("- Risk factors: results/risk_factor_analysis.png")