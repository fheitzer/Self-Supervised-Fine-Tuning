import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_cycle_accs(directory):
    # Initialize a dictionary to store the accuracy values for each model
    accuracy_data = {}
    directory = os.path.join('fine-tuning-data', directory)
    
    # Regex to capture model name and accuracy value
    pattern = re.compile(r'(\w+):\s([0-9.]+)')
    
    # Iterate over the files (assuming the files are named 'cycle_X')
    for file_name in sorted(os.listdir(directory)):
        if file_name.startswith('cycle_'):
            with open(os.path.join(directory, file_name), 'r') as f:
                # Parse each line in the file
                for line in f:
                    match = pattern.match(line)
                    if match:
                        model, accuracy = match.groups()
                        accuracy = float(accuracy)
    
                        # Append accuracy to the model's list
                        if model not in accuracy_data:
                            accuracy_data[model] = []
                        accuracy_data[model].append(accuracy)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy for each model
    for model, accuracies in accuracy_data.items():
        plt.plot(accuracies, label=model, marker='o')
    
    plt.xlabel('Cycle')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Cycles')
    plt.legend()
    plt.grid(True)
    # Save the plot to the same directory
    plot_path = os.path.join(directory, 'model_accuracies_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    

# Function to process all CSVs and return combined DataFrame
def load_csvs(directory):
    dfs = []
    # Loop through all CSV files in the directory
    if not os.path.isdir(directory):
        directory = os.path.dirname(directory)
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            cycle = int(file.split('_')[-1].split('.')[0])  # Extract cycle number from filename
            df = pd.read_csv(os.path.join(directory, file))
            df['cycle'] = cycle  # Add cycle column to each DataFrame
            dfs.append(df)
    # Concatenate all DataFrames
    return pd.concat(dfs, ignore_index=True)


# Function to calculate correctness percentage and positive target ratio
def compute_metrics(df):
    # Group by cycle and compute metrics
    grouped = df.groupby('cycle').apply(lambda x: pd.Series({
        'correctness': np.mean(x['target_true'] == x['target']),  # correctness percentage
        'positive_ratio': np.mean(x['target'] == 1)  # positive ratio percentage
    })).reset_index()

    return grouped


# Function to calculate per-model metrics
def compute_per_model_metrics(df):
    # Group by cycle and model_id to compute metrics
    grouped_per_model = df.groupby(['cycle', 'model_id']).apply(lambda x: pd.Series({
        'correctness': np.mean(x['target_true'] == x['target']) * 100,
        'positive_ratio': np.mean(x['target'] == 1) * 100
    })).reset_index()

    return grouped_per_model


# Function to plot overall correctness and positive ratio
def plot_overall_metrics(metrics):
    plt.figure(figsize=(10, 6))

    # Plot overall correctness
    plt.plot(metrics['cycle'], metrics['correctness'], label='Correctness (%)', color='blue', marker='o')
    # Plot overall positive ratio
    plt.plot(metrics['cycle'], metrics['positive_ratio'], label='Positive Ratio (%)', color='green', marker='o')

    plt.xlabel('Cycle')
    plt.ylabel('Percentage')
    plt.title('Overall Correctness and Positive Target Ratio Across Cycles')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("overall_metrics_plot.png")  # Save the plot in the directory


# Function to plot per-model metrics
def plot_per_model_metrics(metrics):
    plt.figure(figsize=(12, 8))
    models = metrics['model_id'].unique()

    for model_id in models:
        model_data = metrics[metrics['model_id'] == model_id]

        plt.plot(model_data['cycle'], model_data['correctness'], label=f'Correctness {model_id}', marker='o')
        plt.plot(model_data['cycle'], model_data['positive_ratio'], label=f'Positive Ratio {model_id}', marker='x')

    plt.xlabel('Cycle')
    plt.ylabel('Percentage')
    plt.title('Per-Model Correctness and Positive Target Ratio Across Cycles')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("per_model_metrics_plot.png")  # Save the plot in the directory


def plot_cycle_data(collection_name):
    directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")),
                             'datasets',
                             'fine-tuning-data',
                             collection_name
                            )
    
    # Load all CSVs into a single DataFrame
    df = load_csvs(directory)
    
    # Compute overall correctness and positive ratio metrics
    overall_metrics = compute_metrics(df)
    
    # Compute per-model correctness and positive ratio metrics
    per_model_metrics = compute_per_model_metrics(df)
    
    # Plot overall metrics
    plot_overall_metrics(overall_metrics)
    
    # Plot per-model metrics
    plot_per_model_metrics(per_model_metrics)

    
if __name__ == "__main__":
    plot_cycle_data('VIENNA_20240926-2228279')
