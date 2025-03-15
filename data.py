import os
import json
import shutil
import random
import glob
import pandas as pd
import zipfile
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
ZIP_FILENAME = "dataset.zip"  # The filename of the zipped dataset
K_FOLDS = 5                   # Number of folds for cross-validation
DATASET_PATH = "dataset"      # Path within the zip file

def prepare_binary_data_splits(zip_path=ZIP_FILENAME, dataset_path=DATASET_PATH, k_folds=K_FOLDS):
    """
    Create k-fold cross-validation splits for two binary classification tasks:
    1. Alzheimer's (A) vs. Healthy Controls (C)
    2. Alzheimer's (A) vs. Frontotemporal Dementia (F)
    
    The splits are saved to separate JSON files.
    
    Args:
        zip_path: Path to the zip file containing the dataset
        dataset_path: Path within the zip file to the dataset
        k_folds: Number of folds for cross-validation
    """
    # Check if the zip file exists
    if not os.path.exists(zip_path):
        print(f"Zip file {zip_path} not found. Using local dataset.")
        use_zip = False
    else:
        use_zip = True
        
    # Get subject list and their corresponding groups
    if use_zip:
        with zipfile.ZipFile(zip_path, 'r') as z:
            try:
                # Read participants.tsv from zip
                with z.open(f"{dataset_path}/participants.tsv") as f:
                    participants_df = pd.read_csv(f, sep='\t')
                    
                # Get list of subject directories
                subject_dirs = [name for name in z.namelist() 
                               if name.startswith(f"{dataset_path}/sub-") and 
                               name.count('/') == 1]
                subject_ids = [os.path.basename(dir.rstrip('/')) for dir in subject_dirs]
            except Exception as e:
                print(f"Error reading from zip file: {e}")
                return
    else:
        # Read participants.tsv locally
        participants_df = pd.read_csv(f"{dataset_path}/participants.tsv", sep='\t')
        
        # Get list of subject directories
        subject_pattern = os.path.join(dataset_path, "sub-*")
        subject_dirs = glob.glob(subject_pattern)
        subject_ids = [os.path.basename(dir) for dir in subject_dirs]
    
    print(f"Found {len(subject_ids)} subject directories")
    
    # Extract and align subject IDs with their groups
    subjects_with_groups = []
    for subject_id in subject_ids:
        if subject_id in participants_df['participant_id'].values:
            group = participants_df.loc[participants_df['participant_id'] == subject_id, 'Group'].values[0]
            subjects_with_groups.append((subject_id, group))
    
    # Filter subjects for each binary classification task
    a_vs_c_subjects = [(sid, grp) for sid, grp in subjects_with_groups if grp in ['A', 'C']]
    a_vs_f_subjects = [(sid, grp) for sid, grp in subjects_with_groups if grp in ['A', 'F']]
    
    print(f"Alzheimer's vs. Healthy: {len(a_vs_c_subjects)} subjects")
    print(f"Alzheimer's vs. FTD: {len(a_vs_f_subjects)} subjects")
    
    # Process each binary classification task
    for task_name, task_subjects in [
        ('alzheimers_vs_healthy', a_vs_c_subjects),
        ('alzheimers_vs_ftd', a_vs_f_subjects)
    ]:
        # Sort by group to ensure consistent ordering
        task_subjects.sort(key=lambda x: x[1])
        
        # Create arrays for stratified k-fold
        X = np.array([s[0] for s in task_subjects])  # subject IDs
        y = np.array([s[1] for s in task_subjects])  # groups (A, C or A, F)
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Create and save the folds
        fold_data = {}
        
        print(f"\nCreating splits for {task_name}:")
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            train_subjects = X[train_idx].tolist()
            val_subjects = X[val_idx].tolist()
            
            # Count group distribution in this fold
            train_groups = y[train_idx]
            val_groups = y[val_idx]
            
            # Count by group
            if task_name == 'alzheimers_vs_healthy':
                group_keys = ['A', 'C']
            else:  # alzheimers_vs_ftd
                group_keys = ['A', 'F']
                
            train_group_counts = {k: int(sum(train_groups == k)) for k in group_keys}
            val_group_counts = {k: int(sum(val_groups == k)) for k in group_keys}
            
            fold_data[f"fold_{fold_idx+1}"] = {
                'train': train_subjects,
                'validation': val_subjects,
                'train_group_counts': train_group_counts,
                'val_group_counts': val_group_counts
            }
            
            print(f"  Fold {fold_idx+1}:")
            print(f"    Train: {len(train_subjects)} subjects ({train_group_counts})")
            print(f"    Validation: {len(val_subjects)} subjects ({val_group_counts})")
        
        # Save the fold data to a JSON file
        output_file = f"{task_name}_splits.json"
        with open(output_file, 'w') as f:
            json.dump(fold_data, f, indent=2)
        
        print(f"  {task_name} k-fold splits saved to {output_file}")

def zip_dataset(dataset_path, output_zip):
    """
    Zip the dataset folder.
    """
    if os.path.exists(output_zip):
        print(f"Zip file {output_zip} already exists. Skipping zipping.")
        return
    
    print(f"Zipping {dataset_path} to {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(dataset_path))
                zipf.write(file_path, arcname)
    
    print(f"Dataset zipped to {output_zip}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset preparation and k-fold splitting')
    parser.add_argument('--zip', action='store_true', help='Zip the dataset')
    parser.add_argument('--split', action='store_true', help='Create k-fold splits')
    parser.add_argument('--k', type=int, default=K_FOLDS, help='Number of folds')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH, help='Dataset path')
    parser.add_argument('--output', type=str, default=ZIP_FILENAME, help='Output zip filename')
    
    args = parser.parse_args()
    
    if args.zip:
        zip_dataset(args.dataset, args.output)
    
    if args.split or not args.zip:
        prepare_binary_data_splits(args.output, args.dataset, args.k)


# eeg with age
# power/ ratios of power
# csp 
# frequency bands