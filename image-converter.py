# Save this as convert_images.py
from PIL import Image
import os
import sys

def convert_images_to_rgb(input_dir, output_dir):
    """Convert all images in input_dir to RGB format and save to output_dir."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Track conversion stats
    total = 0
    converted = 0
    errors = 0
    
    # Process each image
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            total += 1
            try:
                # Open the image
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path)
                
                # Convert to RGB if not already
                if img.mode != 'RGB':
                    print(f"Converting {filename} from {img.mode} to RGB")
                    img = img.convert('RGB')
                    converted += 1
                else:
                    print(f"{filename} is already in RGB mode")
                
                # Save as JPEG (which is always RGB)
                output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
                img.save(output_path, 'JPEG', quality=95)
                print(f"Saved {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                errors += 1
    
    print(f"\nSummary: {total} images processed, {converted} converted to RGB, {errors} errors")

if __name__ == "__main__":
    # Set default directories
    input_dir = "meloni_images"
    output_dir = "meloni_images_rgb"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print(f"Converting images from {input_dir} to {output_dir}")
    convert_images_to_rgb(input_dir, output_dir)
