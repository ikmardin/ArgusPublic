# ArgusPublic
GEOINT tool for route optimization

Takes in coordinate-stamped imagery, picks out key features and compiles into standardized form.

Plots optimal route from points A to B within a defined area based on these features and a commander's intent

Generates both this path (in red) and the naive shortest distance path (in blue) to illustrate difference

# Setup
Enter file paths for provided imagery, corresponding coordinate stamps, and output file. 
Format of coordinates file is:
{
    File_name : coordinates
}

    e.g.
    
    {

    "Sattelite_Sample1.jpg": "37.533, 47.115",
    "Drone_Sample1.jpg": "37.515015, 47.113348"

    }
    


For the purpose of demonstration, a sample output file has been provided in formattedData/data.json 

Individual sample images have also been provided via Google Maps

# Install dependencies
pip install -r requirements.txt

# Running
streamlit run frontend/app.py 