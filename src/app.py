from inference import run_inference
from utils import save_results

if __name__ == "__main__":
    # Input image and labels
    image_path = "data/images/1733917839.jpeg"
    labels = ["Hoodie", "Jacket", "T-shirt", "Casual", "Black", "White"]

    # Run inference
    results = run_inference(image_path, labels)

    # Save results
    output_path = "data/results/tags.json"
    save_results(results, output_path)

    print(f"Results saved to {output_path}")
