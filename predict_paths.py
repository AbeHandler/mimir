import os
from typing import Dict, List, Tuple
from pathlib import Path
import json

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

def load_config_and_predict(config_path: str) -> None:
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_dir, file_paths = predict_mimir_paths(config)
    
    print(f"Expected base directory:")
    print(f"  {base_dir}")
    print(f"\nExpected result files:")
    for path in sorted(file_paths):
        print(f"  {path}")

# Example usage
if __name__ == "__main__":
    # Load from config file
    config_file = "configs/olmo_blocked_docs.json"
    
    load_config_and_predict(config_file)