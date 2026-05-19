# ==============================
# IMPORT LIBRARIES
# ==============================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.stats import zscore

warnings.filterwarnings("ignore")


# ==============================
# LOAD DATASET
# ==============================

def load_dataset(file_path):

    try:
        df = pd.read_csv(file_path)

        print("\nDataset loaded successfully!")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        return df

    except FileNotFoundError:
        print("\nERROR: Dataset file not found.")
        return None


# ==============================
# INSPECT DATA
# ==============================

def inspect_data(df):

    print("\nFIRST 5 ROWS")
    print(df.head())

    print("\nCOLUMN NAMES")
    print(df.columns)

    print("\nDATA TYPES")
    print(df.dtypes)

    print("\nMISSING VALUES")
    print(df.isnull().sum())


# ==============================
# CLEAN DATA
# ==============================

def clean_data(df):

    df = df.drop_duplicates()
    df = df.dropna()

    df = df[df["Wind Speed (m/s)"] >= 0]
    df = df[df["LV ActivePower (kW)"] >= 0]

    df = df.reset_index(drop=True)

    print("\nData cleaned successfully!")
    print(f"Remaining rows: {len(df)}")

    return df


# ==============================
# SAVE CLEANED DATA
# ==============================

def save_cleaned_data(df):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, "dataset_cleaned.csv")
    df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved: {output_path}")


# ==============================
# DEVIATION COMPUTATION
# ==============================

def compute_deviation(df):

    df["Deviation"] = (
        df["Theoretical_Power_Curve (KWh)"] -
        df["LV ActivePower (kW)"]
    )

    df["Deviation_Percentage"] = (
        df["Deviation"] /
        df["Theoretical_Power_Curve (KWh)"]
    ) * 100

    return df


# ==============================
# WIND LAYERING
# ==============================

def create_wind_layers(df):

    bins = [0, 3, 6, 9, 12, 20]
    labels = ["Very Low", "Low", "Moderate", "High", "Very High"]

    df["Wind Layer"] = pd.cut(df["Wind Speed (m/s)"], bins=bins, labels=labels)

    return df


# ==============================
# MULTI-LAYER ANALYSIS
# ==============================

def layer_performance_analysis(df):

    print("\n=== ADAPTIVE MULTI-LAYER ANALYSIS ===")

    grouped = df.groupby("Wind Layer")[[
        "LV ActivePower (kW)",
        "Theoretical_Power_Curve (KWh)"
    ]].mean()

    grouped["Efficiency (%)"] = (
        grouped["LV ActivePower (kW)"] /
        grouped["Theoretical_Power_Curve (KWh)"]
    ) * 100

    print(grouped)

    return grouped


# ==============================
# UNDERPERFORMANCE CLASSIFICATION
# ==============================

def classify_underperformance(layer_df):

    def label(x):
        if x >= 85:
            return "GOOD"
        elif x >= 60:
            return "MODERATE"
        else:
            return "UNDERPERFORMING"

    layer_df["Performance Status"] = layer_df["Efficiency (%)"].apply(label)

    print("\n=== PERFORMANCE STATUS ===")
    print(layer_df)

    return layer_df


# ==============================
# SEVERITY FILTER (Z-SCORE)
# ==============================

# ==============================
# HIGH-DEVIATION SEVERITY FILTER (THEORY-ALIGNED)
# ==============================

def high_deviation_severity_filter(df):

    import numpy as np
    from scipy.stats import zscore

    # Step 1: Compute Z-score (for anomaly reference only)
    df["Deviation_Z"] = zscore(df["Deviation"])

    # Step 2: Compute statistical parameters (MU and SIGMA)
    mu = np.mean(df["Deviation"])
    sigma = np.std(df["Deviation"])

    print("\n=== STATISTICAL BASELINE ===")
    print("Mean (μ):", mu)
    print("Std Dev (σ):", sigma)

    # Step 3: Classification aligned with your paper
    def severity(d):

        if d <= mu:
            return "LOW"
        elif mu < d <= mu + sigma:
            return "MODERATE"
        elif mu + sigma < d <= mu + 2 * sigma:
            return "SEVERE"
        else:
            return "CRITICAL"

    df["Deviation_Severity"] = df["Deviation"].apply(severity)

    # Step 4: Summary output
    print("\n=== SEVERITY COUNT ===")
    print(df["Deviation_Severity"].value_counts())

    return df


# ==============================
# OUTPUT FOLDER
# ==============================

def create_output_folder():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    return output_dir


# ==============================
# FIGURE FUNCTIONS
# ==============================

def plot_deviation_histogram(df):

    output_dir = create_output_folder()

    plt.figure(figsize=(10, 6))

    plt.hist(df["Deviation"], bins=50, color="navy", edgecolor="black")

    plt.title("Power Curve Deviation Distribution")
    plt.xlabel("Deviation")
    plt.ylabel("Frequency")
    plt.grid(True)

    save_path = os.path.join(output_dir, "figure2_deviation_histogram.png")
    plt.savefig(save_path, dpi=300)

    plt.show()

    print("Saved:", save_path)


def plot_wind_vs_power(df):

    output_dir = create_output_folder()

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["Wind Speed (m/s)"],
        df["LV ActivePower (kW)"],
        c=df["LV ActivePower (kW)"],
        cmap="magma",
        s=10,
        alpha=0.7
    )

    plt.colorbar()

    plt.title("Wind Speed vs Power Output")
    plt.xlabel("Wind Speed")
    plt.ylabel("Power")
    plt.grid(True)

    save_path = os.path.join(output_dir, "figure3_scatter_power.png")
    plt.savefig(save_path, dpi=300)

    plt.show()

    print("Saved:", save_path)


def plot_severity_distribution(df):

    output_dir = create_output_folder()


    counts = df["Deviation_Severity"].value_counts()

    plt.figure(figsize=(8, 5))

    plt.bar(counts.index, counts.values, color=["green", "orange", "red"])

    plt.title("Severity Distribution")
    plt.xlabel("Severity")
    plt.ylabel("Count")
    plt.grid(axis="y")

    save_path = os.path.join(output_dir, "figure4_plot_severity.png")
    plt.savefig(save_path, dpi=300)

    plt.show()

    print("Saved:", save_path)


def plot_deviation_trend(df):

    output_dir = create_output_folder()


    plt.figure(figsize=(10, 6))

    plt.plot(df["Deviation"].values, color="crimson")

    plt.title("Deviation Trend")
    plt.xlabel("Data Points")
    plt.ylabel("Deviation")
    plt.grid(True)

    save_path = os.path.join(output_dir, "figure5_plot_trend.png")
    plt.savefig(save_path, dpi=300)

    plt.show()

    print("Saved:", save_path)

# ==============================
# ANIMATIONS
# ==============================

def animate_power_curve(df):

    output_dir = create_output_folder()
    frames = []
    step = 500

    for i in range(step, len(df), step):

        plt.figure(figsize=(10, 6))

        plt.scatter(
            df["Wind Speed (m/s)"][:i],
            df["LV ActivePower (kW)"][:i],
            c=df["LV ActivePower (kW)"][:i],
            cmap="magma",
            s=10,
            alpha=0.6
        )

        plt.title("Wind Speed vs Power Animation")
        plt.grid(True)

        temp = os.path.join(output_dir, "temp.png")
        plt.savefig(temp)
        plt.close()

        frames.append(imageio.imread(temp))

    imageio.mimsave(
        os.path.join(output_dir, "power_animation.gif"),
        frames,
        duration=0.2
    )


def animate_deviation(df):

    output_dir = create_output_folder()
    frames = []
    step = 500

    for i in range(step, len(df), step):

        plt.figure(figsize=(10, 6))

        plt.plot(df["Deviation"][:i], color="red")

        plt.title("Deviation Animation")
        plt.grid(True)

        temp = os.path.join(output_dir, "temp.png")
        plt.savefig(temp)
        plt.close()

        frames.append(imageio.imread(temp))

    imageio.mimsave(
        os.path.join(output_dir, "deviation_animation.gif"),
        frames,
        duration=0.2
    )


# ==============================
# MAIN PIPELINE
# ==============================

def main():

    dataset_path = r"C:\Users\Stephanie Ericka\OneDrive\Documents\SAM\ComProg_Lab\EDS_TUPM-25-0264_Agunat\data\dataset_original.csv"

    df = load_dataset(dataset_path)
    if df is None:
        return

    inspect_data(df)

    df = clean_data(df)

    df = compute_deviation(df)
    df = create_wind_layers(df)

    layer_df = layer_performance_analysis(df)
    layer_df = classify_underperformance(layer_df)

    df = high_deviation_severity_filter(df)

    save_cleaned_data(df)

    plot_deviation_histogram(df)
    plot_wind_vs_power(df)
    plot_severity_distribution(df)
    plot_deviation_trend(df)

    animate_power_curve(df)
    animate_deviation(df)

    print("\nPipeline executed successfully.")


if __name__ == "__main__":
    main()