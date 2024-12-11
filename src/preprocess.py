from PIL import Image

def preprocess_image(image_path, preprocess):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)
