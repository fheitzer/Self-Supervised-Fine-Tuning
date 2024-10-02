import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib_venn import venn2, venn3

def plot_data_barchart(collection_name):
    """
    Generate bar plots for each CSV file in the directory.
    Each plot shows two bars for each model: one for positive targets and one for negative targets.
    Each bar consists of two parts: correct predictions and total predictions.
    """
    csv_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")),
                             'datasets',
                             'fine-tuning-data',
                             collection_name
                            )
    csv_dir = os.path.dirname(csv_dir)
    # Example usage
    model_names = {
        0: "resnet18",
        1: "resnet34",
        2: "resnet50",
        3: "tf_efficientnet_b0",
        4: "tf_efficientnet_b2",
    }

    # Loop through all CSV files in the directory
    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(csv_dir, csv_file)
            data = pd.read_csv(csv_path)

            # Initialize the bar plot
            plt.figure(figsize=(10, 6))
            model_ids = sorted(data['model_id'].unique())
            bar_width = 0.35
            indices = range(len(model_ids))

            for i, model_id in enumerate(model_ids):
                model_data = data[data['model_id'] == model_id]

                # Separate positive and negative targets
                positive_data = model_data[model_data['target'] == 1]
                negative_data = model_data[model_data['target'] == 0]

                # Count correct predictions (target == target_true) and total predictions
                positive_correct = (positive_data['target'] == positive_data['target_true']).sum()
                positive_total = len(positive_data)

                negative_correct = (negative_data['target'] == negative_data['target_true']).sum()
                negative_total = len(negative_data)

                # Plot bars for positive class
                plt.bar(i - bar_width / 2, positive_correct, color='green', width=bar_width, label='Positive Correct' if i == 0 else "")
                plt.bar(i - bar_width / 2, positive_total, color='black', width=bar_width, bottom=positive_correct, label='Positive Total' if i == 0 else "")

                # Plot bars for negative class
                plt.bar(i + bar_width / 2, negative_correct, color='red', width=bar_width, label='Negative Correct' if i == 0 else "")
                plt.bar(i + bar_width / 2, negative_total, color='black', width=bar_width, bottom=negative_correct, label='Negative Total' if i == 0 else "")

            # Set the labels and title
            plt.xticks(indices, [model_names[model_id] for model_id in indices])
            plt.xlabel('Models')
            plt.ylabel('Number of Datapoints')
            plt.title(f'Predictions for {csv_file}')
            plt.legend()

            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(csv_dir, f"{os.path.splitext(csv_file)[0]}_barplot.png"))
            plt.close()


def load_image_ids(directory, names):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    isic_id_sets = {}
    
    for file in csv_files:
        if file[:-4] in names:
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            isic_ids = set(df['image_id'])
            isic_id_sets[file[:-4]] = isic_ids
    
    return isic_id_sets

def plot_venn(directory, names, i =''):
    isic_id_sets = load_image_ids(directory, names)
    num_csvs = len(isic_id_sets)
    labels = list(isic_id_sets.keys())
    
    if num_csvs == 2:
        set1, set2 = isic_id_sets.values()
        venn = venn2([set1, set2], set_labels=(labels[0], labels[1]))
    elif num_csvs == 3:
        set1, set2, set3 = isic_id_sets.values()
        venn = venn3([set1, set2, set3], set_labels=(labels[0], labels[1], labels[2]))
    else:
        print(f"Currently, this script supports only 2 or 3 CSV files.")
        return
    
    plt.title('Image ID Overlap between CSV Files')
    plt.savefig(f'venn_data_{i}.png')
    plt.clf()

def plot_cycle_accs(directory, name='', data_name='New Clinic'):
    # Initialize dictionaries to store the metrics for each model
    metrics_data = {
        'accuracy': {},
        'balanced_accuracy': {},
        'specificity': {},
        'sensitivity': {}
    }

    directory = os.path.join('fine-tuning-data', directory)
    
    # Regex to capture model name, metric name, and value
    pattern = re.compile(r'(\w+)\s(\w+):\s([0-9.]+)')
    
    # Iterate over the files (assuming the files are named 'cycle_X.txt')
    for file_name in sorted(os.listdir(directory)):
        if file_name.startswith('cycle_'):
            with open(os.path.join(directory, file_name), 'r') as f:
                cycle = int(file_name.split('_')[-1].split('.')[0])  # Extract cycle number from filename
                
                # Parse each line in the file
                for line in f:
                    match = pattern.match(line)
                    if match:
                        model, metric, value = match.groups()
                        value = float(value)
                        
                        # Append the metric value to the corresponding model and metric
                        if model not in metrics_data[metric]:
                            metrics_data[metric][model] = []
                        metrics_data[metric][model].append(value)
    
    # Create a color map for models
    unique_models = metrics_data['accuracy'].keys()
    colors = plt.cm.get_cmap('tab10', len(unique_models))  # Get a color map with enough colors

    # Plot the results
    plt.figure(figsize=(10, 6))
    
    for idx, model in enumerate(unique_models):
        # Assign black color to ensemble model
        model_color = 'black' if model.lower() == 'ensemble' else colors(idx)
        
        # Plot accuracy as a line
        plt.plot(metrics_data['accuracy'][model], label=f'{model} accuracy', marker=None, color=model_color)
        
        # Plot other metrics as points
        plt.scatter(range(len(metrics_data['balanced_accuracy'][model])), 
                    metrics_data['balanced_accuracy'][model], 
                    label=f'{model} balanced accuracy', marker='.', color=model_color)
        
        plt.scatter(range(len(metrics_data['specificity'][model])), 
                    metrics_data['specificity'][model], 
                    label=f'{model} specificity', marker='_', color=model_color)
        
        plt.scatter(range(len(metrics_data['sensitivity'][model])), 
                    metrics_data['sensitivity'][model], 
                    label=f'{model} sensitivity', marker='+', color=model_color)

    plt.xlabel('Cycle')
    plt.ylabel('Percentage')
    plt.title(f'{data_name}: Model Metrics over Cycles')
    
    # Place the legend on the right-hand side
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to the same directory
    plot_path = os.path.join(directory, f"{name}_model_metrics_plot.png")
    plt.savefig(plot_path)
    

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
def plot_overall_metrics(metrics, directory, name='', data_name='New Clinic'):
    plt.figure(figsize=(10, 6))

    # Plot overall correctness
    plt.plot(metrics['cycle'], metrics['correctness'], label='Correctness (%)', color='blue', marker='o')
    # Plot overall positive ratio
    plt.plot(metrics['cycle'], metrics['positive_ratio'], label='Positive Ratio (%)', color='green', marker='o')

    plt.xlabel('Cycle')
    plt.ylabel('Percentage')
    plt.title(f'{data_name}: Overall Correctness and Positive Target Ratio Across Cycles')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    name += 'overall_metrics.png'
    plot_path = os.path.join(os.path.dirname(directory), name)
    plt.savefig(plot_path)  # Save the plot in the directory


# Function to plot per-model metrics
def plot_per_model_metrics(metrics, directory, name='', data_name='New Clinic'):
    plt.figure(figsize=(12, 8))
    models = metrics['model_id'].unique()

    for model_id in models:
        model_data = metrics[metrics['model_id'] == model_id]

        plt.plot(model_data['cycle'], model_data['correctness'], label=f'Correctness {model_id}', marker='o')
        plt.plot(model_data['cycle'], model_data['positive_ratio'], label=f'Positive Ratio {model_id}', marker='x')

    plt.xlabel('Cycle')
    plt.ylabel('Percentage')
    plt.title(f'{data_name}: Per-Model Correctness and Positive Target Ratio Across Cycles')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    name += 'per_model_metrics.png'
    plot_path = os.path.join(os.path.dirname(directory), name)
    plt.savefig(plot_path) # Save the plot in the directory


def plot_cycle_data(collection_name, name, data_name):
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
    plot_overall_metrics(overall_metrics, directory, name, data_name)
    
    # Plot per-model metrics
    plot_per_model_metrics(per_model_metrics, directory, name, data_name)

    
if __name__ == "__main__":
    directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")),
                             'datasets',
                             'fine-tuning-data',
                             'MSK_20241002-070116')
    plot_data_barchart(directory)
    exit()
    
    datasets = [
               'VIENNA',
               'PH2',
               'UDA',
               'SIDNEY',
               'DERM7PT',
               'MSK',
               'HAM10000',
               'BCN',
              ]
    directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'datasets', 'metadata')
    for i, d in enumerate(datasets):
        for j, dd in enumerate(datasets):
            if i == j:
                continue
            plot_venn(directory, [d, dd], d + dd)
