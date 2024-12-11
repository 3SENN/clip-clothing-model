import torch
import clip
from preprocess import preprocess_image
from PIL import Image


def run_inference(image_path, labels):
    import numpy as np
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load and preprocess image
    image = preprocess_image(image_path, preprocess)

    # Tokenize text labels
    text = clip.tokenize(labels).to(device)

    # Run inference
    with torch.no_grad():
        image_features = model.encode_image(image.to(device))
        text_features = model.encode_text(text)

        # Compute probabilities
        logits_per_image = image_features @ text_features.T
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Convert results to standard Python floats
    return {label: float(prob) for label, prob in zip(labels, probs[0])}
