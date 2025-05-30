import os
from typing import Dict, List, Tuple
from pathlib import Path
import json
import csv
import argparse

def predict_mimir_paths(config: Dict) -> Tuple[str, List[str]]:
    experiment_name = config.get("experiment_name", "default_experiment")
    base_model = config.get("base_model", "default_model")
    ourdataset = config.get("ourdataset")
    specific_source = config.get("specific_source")
    blackbox_attacks = config.get("blackbox_attacks", [])
    ref_config = config.get("ref_config", {})
    neighborhood_config = config.get("neighborhood_config", {})
    
    # Default tmp_results path (you may need to adjust this)
    tmp_results = config.get("env_config", {}).get("tmp_results", "tmp_results")
    
    # Build the directory path following MIMIR's logic
    base_model_name = base_model.replace("/", "_")
    
    # Start with experiment name and model
    path_parts = [tmp_results, experiment_name, base_model_name]
    
    # Add dataset-specific path
    if ourdataset is not None:
        path_parts.append(ourdataset)  # Keep slashes intact
    elif specific_source is not None:
        # Process specific source name (simplified version)
        processed_source = specific_source.replace("<", "").replace(">", "").replace("_", "-")
        path_parts.append(processed_source)
    
    base_directory = os.path.join(*path_parts)
    
    # Predict expected result files (only _results.json files)
    expected_files = []
    
    # Attack-specific result files
    for attack in blackbox_attacks:
        if attack == "ref":
            # Reference-based attacks create files based on reference models
            ref_models = ref_config.get("models", [])
            for model in ref_models:
                model_name = model.split("/")[-1]  # Get just the model name part
                expected_files.append(f"ref-{model_name}_results.json")
        elif attack == "ne":
            # Neighborhood attack files based on perturbation list
            n_perturbation_list = neighborhood_config.get("n_perturbation_list", [5])
            for n_pert in n_perturbation_list:
                expected_files.append(f"ne-{n_pert}_results.json")
        else:
            # Standard attacks (loss, min_k, zlib, etc.)
            expected_files.append(f"{attack}_results.json")
    
    # Convert to full paths
    full_file_paths = [os.path.join(base_directory, f) for f in expected_files]
    
    return base_directory, full_file_paths

def validate_paths(base_directory: str, file_paths: List[str]) -> None:
    """Validate that the predicted paths exist, throw errors if not."""
    # Check if base directory exists
    if not os.path.exists(base_directory):
        raise FileNotFoundError(f"Base directory does not exist: {base_directory}")
    
    if not os.path.isdir(base_directory):
        raise NotADirectoryError(f"Path exists but is not a directory: {base_directory}")
    
    # Check each expected file
    missing_files = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        error_msg = f"Expected result files do not exist:\n"
        for missing_file in missing_files:
            error_msg += f"  - {missing_file}\n"
        raise FileNotFoundError(error_msg.rstrip())

def flatten_results_to_csv(file_paths: List[str], output_csv: str = "mimir_results.csv") -> None:
    """
    Flatten MIMIR result files to CSV format.
    Each file should have an 'id_to_score' field with 'member' and 'nonmember' keys.
    """
    csv_rows = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: Skipping missing file: {file_path}")
            continue
            
        # Extract method name from filename (remove _results.json)
        filename = os.path.basename(file_path)
        method_name = filename.replace("_results.json", "")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            id_to_score = data.get('id_to_score', {})
            
            # Process member scores
            member_scores = id_to_score.get('member', {})
            for doc_id, score in member_scores.items():
                csv_rows.append({
                    'method': method_name,
                    'membership': 'member',
                    'doc_id': doc_id,
                    'score': score
                })
            
            # Process nonmember scores
            nonmember_scores = id_to_score.get('nonmember', {})
            for doc_id, score in nonmember_scores.items():
                csv_rows.append({
                    'method': method_name,
                    'membership': 'nonmember',
                    'doc_id': doc_id,
                    'score': score
                })
                
            print(f"Processed {len(member_scores)} member and {len(nonmember_scores)} nonmember scores from {filename}")
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Write to CSV
    if csv_rows:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['method', 'membership', 'doc_id', 'score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"\n✅ Successfully wrote {len(csv_rows)} rows to {output_csv}")
    else:
        print(f"\n❌ No data to write to CSV")

def load_config_and_predict(config_path: str, flatten_to_csv: bool = False, csv_output: str = "mimir_results.csv") -> None:
    # Check if config file exists first
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_dir, file_paths = predict_mimir_paths(config)
    
    print(f"Expected base directory:")
    print(f"  {base_dir}")
    print(f"\nExpected result files:")
    for path in sorted(file_paths):
        print(f"  {path}")
    
    # Validate that all paths exist
    try:
        validate_paths(base_dir, file_paths)
        print(f"\n✅ All paths exist successfully!")
        
        # Optionally flatten results to CSV
        if flatten_to_csv:
            print(f"\nFlattening results to CSV...")
            flatten_results_to_csv(file_paths, csv_output)
            
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"\n❌ Path validation failed:")
        print(f"  {str(e)}")
        
        # Still try to flatten existing files if requested
        if flatten_to_csv:
            print(f"\nAttempting to flatten existing files to CSV...")
            flatten_results_to_csv(file_paths, csv_output)
        
        raise

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="e.g. olmo_blocked_docs.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_file = f"configs/{args.config}.json"
    
    try:
        load_config_and_predict(config_file, flatten_to_csv=True, csv_output=f"{args.config}.csv")
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        exit(1)