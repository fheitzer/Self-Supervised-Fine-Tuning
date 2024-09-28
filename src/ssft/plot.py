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

def plot_cycle_data(directory):

    directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'datasets', 'fine-tuning-data', directory)
    # Load the CSV data
    df = pd.read_csv(directory + '.csv')
    
    
    # Add a column for correct/incorrect classification
    df['correct'] = df['target_true'] == df['target']
    
    # Group the data by model_id, target (class), and whether it was correct
    grouped = df.groupby(['model_id', 'target', 'correct']).size().reset_index(name='count')
    
    # Pivot the table to separate correct and incorrect counts
    pivot_table = grouped.pivot_table(index=['model_id', 'target'], columns='correct', values='count', fill_value=0)
    
    # Separate correct and incorrect counts
    correct_classifications = pivot_table[True] if True in pivot_table.columns else pd.DataFrame()
    incorrect_classifications = pivot_table[False] if False in pivot_table.columns else pd.DataFrame()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Unstack correct classifications for plotting (reshape to have classes as columns)
    correct_unstacked = correct_classifications.unstack(level='target')
    
    # Plot correct classifications (colored bars)
    correct_unstacked.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
    
    # Overlay incorrect classifications (black bars)
    # For each model_id and target, we plot the black bars
    for idx, (model, target) in enumerate(correct_classifications.index):
        correct_count = correct_classifications.loc[(model, target)] if (model, target) in correct_classifications.index else 0
        incorrect_count = incorrect_classifications.loc[(model, target)] if (model, target) in incorrect_classifications.index else 0
        ax.bar(idx, incorrect_count, bottom=correct_count, width=0.8, color='black')
    
    # Set plot title and labels
    plt.title('Amount of Collected Data (Correct vs Incorrect) per Model and Class')
    plt.xlabel('Model')
    plt.ylabel('Number of Data Points')
    
    # Display legend (classes)
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show the plot
    plt.tight_layout()
    plot_path = directory + '_plot.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    

if __name__ == "__main__":
    plot_cycle_data('VIENNA_20240926-2228279')
