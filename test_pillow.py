from PIL import Image

# Open an image file
img_path = 'passport.png'  # Replace with an actual image path
with Image.open(img_path) as img:
    print(f"Image opened successfully: {img.format}, {img.size}, {img.mode}")
