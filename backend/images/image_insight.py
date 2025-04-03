# Contains image analysis logic

import os
import json
import base64
from openai import OpenAI

API_KEY = ""  


# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# Define paths
IMAGE_FOLDER = ""  # Folder containing JPEG images e.g. ./.../backend/images
COORDINATES_FILE = ""  # Coordinates data e.g. ./.../backend/images/coordinates.json
OUTPUT_FILE = ""  # Output file

def process_coordinates_file(file_path):
    """
    Process coordinates file, keeping only the first occurrence of each image.
    Returns a dictionary mapping image filenames to their coordinates.
    """
    try:
        with open(file_path, 'r') as f:
            coordinates_data = json.load(f)
        
        # Track which images we've seen to handle duplicates
        processed_images = {}
        skipped_duplicates = []
        
        for image_file, coords in coordinates_data.items():
            if image_file in processed_images:
                skipped_duplicates.append(image_file)
            else:
                processed_images[image_file] = coords
        
        if skipped_duplicates:
            print(f"INFO: Found duplicate entries for these images (using first occurrence only): {', '.join(skipped_duplicates)}")
            
        return processed_images
        
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in coordinates file: {e}")
        return {}
    except Exception as e:
        print(f"ERROR: Could not process coordinates file: {e}")
        return {}

def main():
    # Initialize OpenAI client
    client = OpenAI(api_key=API_KEY)
    
    # Process coordinates file, skipping duplicates
    print(f"Reading coordinates from: {COORDINATES_FILE}")
    coordinates_data = process_coordinates_file(COORDINATES_FILE)
    
    if not coordinates_data:
        print("No valid coordinates found. Exiting.")
        return
    
    # Find all images
    print(f"Looking for images in folder: {IMAGE_FOLDER}")
    if not os.path.exists(IMAGE_FOLDER):
        print(f"ERROR: Image folder not found: {IMAGE_FOLDER}")
        return
        
    image_files = [f for f in os.listdir(IMAGE_FOLDER) 
                   if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not image_files:
        print("No JPEG images found in the specified folder.")
        return
        
    print(f"Found {len(image_files)} JPEG images.")
    
    results = []
    processed_count = 0
    
    # Track which coordinates we've used (for information only)
    used_coordinates = set()
    
    # Process each image
    for image_file in image_files:
        if image_file not in coordinates_data:
            print(f"Skipping {image_file}: no coordinates found")
            continue
        
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        coordinates = coordinates_data[image_file]
        
        # Note if coordinates are used multiple times (just informational)
        if coordinates in used_coordinates:
            print(f"Note: Coordinates {coordinates} are being used for multiple images (now {image_file})")
        used_coordinates.add(coordinates)
        
        print(f"Processing {image_file} with coordinates {coordinates}...")
        
        # Read and encode image
        try:
            with open(image_path, "rb") as img:
                base64_image = base64.b64encode(img.read()).decode('utf-8')
        except Exception as e:
            print(f"Error reading image {image_file}: {e}")
            continue
        
        # System prompt for analysis
        SYSTEM_PROMPT = """Analyze this satellite/aerial image and provide a detailed tactical description. 
                        Focus on:
                        - Enemy presence and installations
                        - Terrain features and obstacles
                        - Roads and infrastructure condition
                        - Strategic advantages or risks

                        Provide the information in a concise paragraph format similar to a military intelligence report.
                        Be specific with details like approximate measurements, percentages, and tactical significance."""
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]}
                ],
                max_tokens=300
            )
            
            # Get description from response
            description = response.choices[0].message.content.strip()
            
            # Add to results
            results.append({
                "coordinates": coordinates,
                "description": description
            })
            
            processed_count += 1
            print(f"Successfully processed {image_file}")
            
        except Exception as e:
            print(f"Error processing {image_file} with API: {str(e)}")
    
    if processed_count == 0:
        print("No images were successfully processed.")
        return
    
    # Save results to JSON
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Analysis complete. Processed {processed_count} images.")
        print(f"Results saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving results to {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    main()