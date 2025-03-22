import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the standardized data
df = pd.read_csv("../data/processed/standardized_data.csv")

# Set plot style
sns.set_theme(style="whitegrid")

# Plot distributions for Age, Cholesterol, and Blood Pressure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Age Distribution
sns.histplot(df["Age"], kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Distribution of Age (Standardized)")
axes[0].set_xlabel("Age (Z-score)")

# Cholesterol Distribution
sns.histplot(df["Cholesterol"], kde=True, ax=axes[1], color="salmon")
axes[1].set_title("Distribution of Cholesterol (Standardized)")
axes[1].set_xlabel("Cholesterol (Z-score)")

# Blood Pressure Distribution
sns.histplot(df["Blood Pressure"], kde=True, ax=axes[2], color="lightgreen")
axes[2].set_title("Distribution of Blood Pressure (Standardized)")
axes[2].set_xlabel("Blood Pressure (Z-score)")

plt.tight_layout()
plt.savefig("results/feature_distributions.png")  # Save to results folder
plt.show()

# Pie chart for Heart Disease class balance
heart_disease_counts = df["Heart Disease"].value_counts()
plt.pie(
    heart_disease_counts,
    labels=["No Heart Disease", "Heart Disease"],
    autopct="%1.1f%%",
    colors=["lightblue", "lightcoral"],
    explode=(0.1, 0),
)
plt.title("Heart Disease Class Distribution")
plt.savefig("results/class_balance.png")  # Save to results folder
plt.show()