import json

def save_results(results, output_path):
    import json
    # Save the results directly as JSON without attempting to convert to float
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
