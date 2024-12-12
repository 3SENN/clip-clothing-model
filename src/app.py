from inference import run_inference
from utils import save_results

if __name__ == "__main__":
    # Input image
    image_path = "data/images/1733917839.jpeg"

    # Tags categorized
    categories = {
        "Clothing Types": [
            "Casual Hoodie", "Oversized Hoodie", "Zipper Hoodie", "Formal Jacket", "Leather Jacket",
            "Denim Jacket", "Bomber Jacket", "Windproof Jacket", "Plain T-shirt", "Graphic T-shirt",
            "Long-sleeve T-shirt", "Crewneck T-shirt", "Slim-fit Jeans", "Ripped Jeans",
            "High-waisted Jeans", "Denim Shorts", "Casual Dress", "Maxi Dress", "Summer Dress",
            "Formal Gown", "Pleated Skirt", "Mini Skirt", "Pencil Skirt", "Flared Skirt"
        ],
        "Styles": [
            "Modern Streetwear", "Vintage Casual", "Minimalist Look", "Classic Style",
            "Timeless Elegance", "High-end Designer Wear", "Professional Office Attire",
            "Athletic Activewear", "Sporty Casual", "Utility Functional Clothing",
            "Bohemian Festival Style", "Bold Statement Look", "Relaxed and Comfy", "Luxury Chic",
            "Retro-inspired Fashion"
        ],
        "Fits": [
            "Relaxed Loose Fit", "Oversized Baggy Fit", "Regular True-to-size Fit",
            "Tailored Custom Fit", "Slim Fit Silhouette", "Stretchable Elastic Fit",
            "Wrinkle-resistant Fabric", "Cropped Above-waist Fit", "High-rise Waist-accentuating Fit",
            "Low-rise Hip Fit", "Layered Outfit-ready Fit"
        ],
        "Fabrics": [
            "Soft Cotton", "Lightweight Linen", "Durable Polyester", "Denim Fabric",
            "Warm Wool Blend", "Smooth Silk Fabric", "Shiny Satin", "Soft Velvet Texture",
            "Stretchy Spandex", "Breathable Mesh", "Moisture-wicking Quick-dry Material",
            "Fluffy Faux Fur", "Luxurious Cashmere", "Insulated Thermal Fabric", "Rainproof Waterproof Material",
            "Cozy Fleece-lined Fabric", "Soft Suede", "Corduroy Ribbed Fabric", "Genuine Leather Material"
        ],
        "Colors": [
            "Solid Black", "Pure White", "Charcoal Gray", "Neutral Beige", "Light Pastel Shades",
            "Bright Neon Colors", "Bold and Vibrant", "Dark Muted Tones", "Floral Pattern",
            "Striped Horizontal Lines", "Checkered Plaid", "Polka Dot Spots", "Animal Print Pattern",
            "Gradient Ombre Shades", "Psychedelic Tie-dye Swirls", "Shiny Metallic Gold",
            "Glittery Sparkle Finish", "Glossy Reflective Surface", "Matte Subtle Finish",
            "Deep Maroon Red", "Teal Blue-green", "Mustard Yellow", "Earthy Olive Green",
            "Navy Deep Blue", "Soft Lavender Purple"
        ],
        "Occasions": [
            "Casual Everyday Wear", "Weekend Outing Outfit", "Shopping-friendly Attire",
            "Office Workwear", "Formal Business Suit", "Gala Event Gown",
            "Wedding Celebration Outfit", "Gym Activewear", "Yoga Stretchable Clothes",
            "Running Lightweight Gear", "Outdoor Sports Functional Outfit",
            "Hiking Durable Clothing", "Camping Gear", "Adventure-ready Outfit",
            "Beachwear Summer Clothes", "Party Festive Attire", "Festival Boho Wear",
            "Holiday Vacation Clothes", "Cozy Sleepwear Pajamas", "Loungewear Relaxed Home Outfit",
            "Work Uniform", "Lab Coat Professional", "Medical Scrubs", "Travel Wrinkle-free Outfit",
            "Long-haul Comfortable Clothing", "Wedding Guest Formal Wear", "Concert Bold Outfit",
            "Team Sports Fan Jersey"
        ]
    }

    # Flatten all tags for inference
    all_tags = [tag for tags in categories.values() for tag in tags]

    # Run inference with the CLIP model
    all_results = run_inference(image_path, all_tags)

    # Group results by category and find the highest confidence tag for each
    filtered_results = {}
    for category, tags in categories.items():
        category_results = {tag: all_results[tag] for tag in tags if tag in all_results}
        if category_results:  # Ensure category has valid results
            highest_confidence_tag = max(category_results, key=category_results.get)
            filtered_results[category] = {
                "tag": highest_confidence_tag,
                "confidence": category_results[highest_confidence_tag]
            }

    # Save results
    output_path = "data/results/tags_by_category.json"
    save_results(filtered_results, output_path)

    print(f"Category-wise highest confidence tags saved to {output_path}")
