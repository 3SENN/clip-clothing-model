import json

def save_results(results, output_path):
    # Convert float32 values to Python float
    results = {key: float(value) for key, value in results.items()}
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
