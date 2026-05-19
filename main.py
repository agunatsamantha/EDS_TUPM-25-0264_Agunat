# ==============================
# IMPORT LIBRARIES
# ==============================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio.v2 as imageio
from scipy.stats import zscore

warnings.filterwarnings("ignore")

# ==============================
# LOAD DATASET
# ==============================

def load_dataset(file_path):
    """
    Loads the wind turbine dataset safely.
    """

    try:
        df = pd.read_csv(file_path)

        print("\nDataset loaded successfully!")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        return df

    except FileNotFoundError:
        print("\nERROR: Dataset file not found.")
        return None

    except pd.errors.EmptyDataError:
        print("\nERROR: Dataset is empty.")
        return None

    except Exception as e:
        print(f"\nUnexpected Error: {e}")
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

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove rows with missing values
    df = df.dropna()

    # Remove negative wind speed
    df = df[df["Wind Speed (m/s)"] >= 0]

    # Remove negative power output
    df = df[df["LV ActivePower (kW)"] >= 0]

    # Reset index
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

    print(f"\nCleaned dataset saved to: {output_path}")

# ==============================
# DESCRIPTIVE ANALYSIS
# ==============================

def descriptive_analysis(df):

    print("\n=== DESCRIPTIVE ANALYSIS ===")

    wind_speed = df["Wind Speed (m/s)"]
    actual_power = df["LV ActivePower (kW)"]
    theoretical_power = df["Theoretical_Power_Curve (KWh)"]

    print("\nWind Speed Stats:")
    print("Mean:", np.mean(wind_speed))
    print("Std:", np.std(wind_speed))
    print("Min:", np.min(wind_speed))
    print("Max:", np.max(wind_speed))

    print("\nActual Power Stats:")
    print("Mean:", np.mean(actual_power))
    print("Std:", np.std(actual_power))

    print("\nTheoretical Power Stats:")
    print("Mean:", np.mean(theoretical_power))
    print("Std:", np.std(theoretical_power))

# ==============================
# POWER CURVE DEVIATION
# ==============================

def compute_deviation(df):

    df["Deviation"] = df["Theoretical_Power_Curve (KWh)"] - df["LV ActivePower (kW)"]

    df["Deviation_Percentage"] = (df["Deviation"] / df["Theoretical_Power_Curve (KWh)"]) * 100

    return df

# ==============================
# WIND SPEED LAYERING
# ==============================

def create_wind_layers(df):

    bins = [0, 3, 6, 9, 12, 20]
    labels = ["Very Low", "Low", "Moderate", "High", "Very High"]

    df["Wind Layer"] = pd.cut(df["Wind Speed (m/s)"], bins=bins, labels=labels)

    return df

# ==============================
# LAYER PERFORMANCE ANALYSIS
# ==============================

def layer_performance_analysis(df):

    print("\n=== ADAPTIVE MULTI-LAYER ANALYSIS ===")

    grouped = df.groupby("Wind Layer")[["LV ActivePower (kW)", "Theoretical_Power_Curve (KWh)"]].mean()

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

    def label_efficiency(x):
        if x >= 85:
            return "GOOD"
        elif x >= 60:
            return "MODERATE"
        else:
            return "UNDERPERFORMING"

    layer_df["Performance Status"] = layer_df["Efficiency (%)"].apply(label_efficiency)

    print("\n=== PERFORMANCE STATUS ===")
    print(layer_df[["Efficiency (%)", "Performance Status"]])

    return layer_df

# ==============================
# HIGH-DEVIATION SEVERITY FILTER
# ==============================

def high_deviation_severity_filter(df):

    # Compute Z-scores for deviation
    df["Deviation_Z"] = zscore(df["Deviation"])

    # Classification function
    def severity(z):
        if abs(z) < 1:
            return "NORMAL"
        elif abs(z) < 2:
            return "MODERATE"
        else:
            return "CRITICAL"

    df["Deviation_Severity"] = df["Deviation_Z"].apply(severity)

    print("\n=== HIGH DEVIATION SEVERITY ===")
    print(df["Deviation_Severity"].value_counts())

    return df

# ==============================
# OUTPUT SETUP
# ==============================

def create_output_folder():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    return output_dir

# ==============================
# GRAPH 1: POWER CURVE
# ==============================

def plot_power_curve(df):

    create_output_folder()

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["Wind Speed (m/s)"],
        df["LV ActivePower (kW)"],
        s=10,
        alpha=0.5
    )

    plt.title("Wind Speed vs Actual Power (Power Curve)")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Actual Power (kW)")
    plt.grid(True)

    plt.savefig("outputs/power_curve.png")
    plt.show()

# ==============================
# GRAPH 2: ACTUAL VS THEORETICAL POWER
# ==============================

def plot_actual_vs_theoretical(df):

    create_output_folder()

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["Wind Speed (m/s)"],
        df["LV ActivePower (kW)"],
        label="Actual Power",
        s=10,
        alpha=0.5
    )

    plt.scatter(
        df["Wind Speed (m/s)"],
        df["Theoretical_Power_Curve (KWh)"],
        label="Theoretical Power",
        s=10,
        alpha=0.5
    )

    plt.title("Actual vs Theoretical Power Curve")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True)

    plt.savefig("outputs/actual_vs_theoretical.png")
    plt.show()

# ==============================
# GRAPH 3: DEVIATION DISTRIBUTION
# ==============================

def plot_deviation_distribution(df):

    create_output_folder()

    plt.figure(figsize=(10, 6))

    plt.hist(df["Deviation"], bins=50, color="blue", alpha=0.7)

    plt.title("Distribution of Power Curve Deviation")
    plt.xlabel("Deviation (Theoretical - Actual)")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.savefig("outputs/deviation_distribution.png")
    plt.show()

# ==============================
# ANIMATION 1: POWER EVOLUTION
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
            s=10,
            alpha=0.5
        )

        plt.title("Wind Speed vs Power (Animated)")
        plt.xlabel("Wind Speed (m/s)")
        plt.ylabel("Power (kW)")
        plt.grid(True)

        temp_path = os.path.join(output_dir, "temp.png")
        plt.savefig(temp_path)
        plt.close()

        frames.append(imageio.imread(temp_path))

    gif_path = os.path.join(output_dir, "power_animation.gif")
    imageio.mimsave(gif_path, frames, duration=0.2)

    print("Saved:", gif_path)

# ==============================
# ANIMATION 2: DEVIATION EVOLUTION
# ==============================

def animate_deviation(df):

    output_dir = create_output_folder()

    frames = []
    step = 500

    for i in range(step, len(df), step):

        plt.figure(figsize=(10, 6))

        plt.plot(df["Deviation"][:i], color="red")

        plt.title("Power Deviation Over Time")
        plt.xlabel("Data Points")
        plt.ylabel("Deviation")
        plt.grid(True)

        temp_path = os.path.join(output_dir, "temp.png")
        plt.savefig(temp_path)
        plt.close()

        frames.append(imageio.imread(temp_path))

    gif_path = os.path.join(output_dir, "deviation_animation.gif")
    imageio.mimsave(gif_path, frames, duration=0.2)

    print("Saved:", gif_path)

# ==============================
# MAIN FUNCTION
# ==============================

def main():
    dataset_path = r"C:\Users\Stephanie Ericka\OneDrive\Documents\SAM\ComProg_Lab\EDS_TUPM-25-0264_Agunat\data\dataset_original.csv"
    df = load_dataset(dataset_path)

    if df is None:
        return

    # Inspect dataset
    inspect_data(df)

    # Clean dataset
    df = clean_data(df)

    # Deviation
    df = compute_deviation(df)

    # Adaptive Multi-Layer Underperformance Filter
    df = create_wind_layers(df)
    layer_df = layer_performance_analysis(df)
    layer_df = classify_underperformance(layer_df)

    # Severity Filter
    df = high_deviation_severity_filter(df)

    # Save cleaned dataset
    save_cleaned_data(df)

    # Descriptive analysis
    descriptive_analysis(df)

    # Static Graphs
    plot_power_curve(df)
    plot_actual_vs_theoretical(df)
    plot_deviation_distribution(df)

    # Animated Graphs
    animate_power_curve(df)
    animate_deviation(df)


# Run program
if __name__ == "__main__":
    main()

