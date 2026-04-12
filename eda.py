import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from config import TRAIN_DIR, TEST_DIR, VALID_DIR


def extract_labels_from_folder(folder_path):
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            label = filename.split('_')[0]
            labels.append(label)
    return labels


def run_eda():
    train_labels = extract_labels_from_folder(TRAIN_DIR)
    test_labels = extract_labels_from_folder(TEST_DIR)
    val_labels = extract_labels_from_folder(VALID_DIR)

    train_counter = Counter(train_labels)
    test_counter = Counter(test_labels)
    val_counter = Counter(val_labels)

    labels = sorted(set(train_labels + test_labels + val_labels))
    label_summary = pd.DataFrame({
        'Train': [train_counter.get(label, 0) for label in labels],
        'Test': [test_counter.get(label, 0) for label in labels],
        'Validation': [val_counter.get(label, 0) for label in labels],
    }, index=labels)
    label_summary['Total'] = label_summary.sum(axis=1)

    print(label_summary)

    contingency_table = label_summary[['Train', 'Test', 'Validation']].T.values
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f'Chi-Square Test Statistic: {chi2}')
    print(f'P-Value: {p_value}')
    print(f'Degrees of Freedom: {dof}')
    print(f'Expected Frequencies:\n{expected}')

    label_summary[['Train', 'Test', 'Validation']].plot(kind='bar', figsize=(10, 5))
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_eda()
