"""
Simple EDA for Heart Disease dataset
Run: python eda.py
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

RAW_PATH = "data/raw/heart_disease_raw.csv"
PROC_PATH = "data/processed/heart_disease_clean_binary.csv"

def basic_info(df, name="df"):
    print(f"\n===== {name} HEAD =====")
    print(df.head())
    print(f"\n===== {name} INFO =====")
    print(df.info())
    print(f"\n===== {name} DESCRIBE =====")
    print(df.describe())
    print(f"\n===== {name} NA COUNTS =====")
    print(df.isna().sum())

def plot_target_distribution(df, target_col, title_suffix=""):
    sns.countplot(x=target_col, data=df)
    plt.title(f"Target Distribution {title_suffix}")
    plt.savefig(f"logs/target_distribution{title_suffix}.png", dpi=120, bbox_inches="tight")
    plt.close()

def plot_corr_heatmap(df, title_suffix=""):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
    plt.title(f"Correlation Heatmap {title_suffix}")
    plt.tight_layout()
    plt.savefig(f"logs/corr_heatmap{title_suffix}.png", dpi=120)
    plt.close()

def main():
    Path("logs").mkdir(exist_ok=True)

    # Raw data
    df_raw = pd.read_csv(RAW_PATH)
    basic_info(df_raw, "RAW")
    raw_target = df_raw.columns[-1]
    print("\nRaw target value counts:")
    print(df_raw[raw_target].value_counts())

    plot_target_distribution(df_raw, raw_target, "_raw")

    # Processed binary data
    df_bin = pd.read_csv(PROC_PATH)
    basic_info(df_bin, "BINARY")
    print("\nBinary target value counts:")
    print(df_bin["target_binary"].value_counts())

    plot_target_distribution(df_bin, "target_binary", "_binary")
    plot_corr_heatmap(df_bin.drop(columns=[raw_target]), "_binary")

if __name__ == "__main__":
    main()
