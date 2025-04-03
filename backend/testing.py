# Contains underlying votes/weightage/route planning logic

import random
import json
import osmnx as ox
from geopy.distance import distance
from shapely.geometry import Polygon, LineString, box
import geopandas as gpd
from openai import OpenAI
import matplotlib.pyplot as plt
import contextily as ctx
import networkx as nx
import numpy as np

# insert openAI api key
API_KEY = ""

DATA_FILE_PATH = "" # Path to output data; a sample is provided in formattedData/data.json

DEFAULT_WEIGHT = 25000 # Change this to change default weighting of empty boxes as a measure of how hostile an
                       # are is in general

def generate_small_boxes(location_point, box_size=100):
    """
    Generate a dictionary of bounding boxes (as shapely Polygons) around a central point.
    
    Parameters:
        location_point (tuple): (lat, lon) central coordinate.
        box_size (int): approximate size of each small box in meters.
        
    Returns:
        dict: keys are tuples (north, south, east, west) and values are Polygon objects.
    """
    # OSMnx returns (west, south, east, north) 
    w, s, e, n = ox.utils_geo.bbox_from_point(location_point, dist=2000, project_utm=False)
    # Reorder so that north, south, east, and west are correctly assigned.
    north, south, east, west = float(n), float(s), float(e), float(w)
    print("Bounding Box from center:", north, south, east, west)
    
    # Calculate latitude and longitude steps using geopy distances.
    lat_step = distance(meters=box_size).destination(point=(south, west), bearing=0)[0] - south
    lon_step = distance(meters=box_size).destination(point=(south, west), bearing=90)[1] - west

    boxes_dict = {}
    current_north = north
    while current_north > south:
        current_south = current_north - lat_step
        current_west = west
        while current_west < east:
            current_east = current_west + lon_step
            polygon = Polygon([
                (current_west, current_north), 
                (current_east, current_north),
                (current_east, current_south), 
                (current_west, current_south)
            ])
            boxes_dict[(current_north, current_south, current_east, current_west)] = polygon
            current_west += lon_step
        current_north -= lat_step
    # print(boxes_dict)
    return boxes_dict


def in_bbox(bbox, lat, lon):
    """
    Check if a given (lat, lon) point lies within the bounding box.
    
    Parameters:
        bbox (tuple): (north, south, east, west)
        lat (float): latitude of the point.
        lon (float): longitude of the point.
        
    Returns:
        bool: True if the point is within the bbox, False otherwise.
    """
    north, south, east, west = bbox
    return (lat <= north) and (lat >= south) and (lon <= east) and (lon >= west)


def assign_features_to_boxes(json_data, boxes):
    """
    Assign features (with coordinates and descriptions) from JSON data to the corresponding bounding boxes.
    
    Parameters:
        json_data (list): list of feature dictionaries. Each dictionary must have:
                          - 'coordinates': string in format "lon, lat"
                          - 'description': textual description of the feature.
        boxes (dict): keys are bbox tuples and values are dicts where features will be added.
    """
    for feature in json_data:
        # Convert coordinates from string "lon, lat" to floats
        coords = [float(x.strip()) for x in feature['coordinates'].split(',')]
        lon, lat = coords
        desc = feature['description']
        for bbox in boxes:
            if in_bbox(bbox, lat, lon):
                if 'features' not in boxes[bbox]:
                    boxes[bbox]['features'] = []
                boxes[bbox]['features'].append({'coords': (lat, lon), 'description': desc})
        print(boxes)


def parse_votes_str(votes_str):
    """
    Parse the votes string (JSON formatted) into a dictionary with bbox tuples as keys and integer votes.
    
    Parameters:
        votes_str (str): JSON string representing vote distribution.
        
    Returns:
        dict: keys are bbox tuples and values are integer vote counts.
    """    
    try:
        raw_dict = json.loads(votes_str)
        votes_dict = {}
        for key, value in raw_dict.items():
            # Convert string key like "(north, south, east, west)" into a tuple of floats.
            tuple_key = tuple(map(float, key.strip('()').split(',')))
            int_value = int(value)
            votes_dict[tuple_key] = int_value
        return votes_dict
    except (json.JSONDecodeError, ValueError) as e:
        print("Error parsing votes:", e)
        return {}


def calculate_weight(G, votes_dict):
    """
    Adjust edge weights in the graph G based on the minimum vote count among intersecting bounding boxes.
    
    Parameters:
        G (networkx.MultiDiGraph): road network graph.
        votes_dict (dict): dictionary with bbox tuples as keys and votes as values.
    """
    # Ensure each edge has a geometry.
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'geometry' not in data:
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            data['geometry'] = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
    # Adjust weight based on vote distribution.
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_geometry = data['geometry']
        min_votes = float('inf')
        for bbox_coords, votes in votes_dict.items():
            north, south, east, west = bbox_coords
            bbox_polygon = box(west, south, east, north)
            if edge_geometry.intersects(bbox_polygon) or bbox_polygon.contains(edge_geometry):
                min_votes = min(min_votes, votes)
        if min_votes == float('inf'):
            min_votes = 1
        edge_length = data['length']
        # Incorporate length as well as votes into the final weight
        data['weight'] = edge_length / min_votes 


def plot_weighted_graph(G, weight_method='weight', path=None, title='Weighted Graph'):
    """
    Plot a graph with edges colored according to their weights.
    
    Parameters:
    -----------
    G : networkx.MultiDiGraph
        The input graph to plot
    weight_method : str
        The edge attribute to use for weights
    path : list, optional
        List of nodes representing a path to highlight
    title : str
        Title for the plot
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Get all edge weights (handling MultiDiGraph)
    edge_weights = []
    edge_colors = []
    
    for u, v, key, data in G.edges(keys=True, data=True):
        weight = data.get(weight_method, 1)
        edge_weights.append(weight)
    
    # Create a color normalization
    norm = plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    
    # Create colors (using YlOrRd colormap - yellow for low weights, red for high)
    for weight in edge_weights:
        color = plt.cm.YlOrRd(norm(weight))
        # Convert RGBA to hex
        hex_color = '#%02x%02x%02x' % tuple(int(255*c) for c in color[:3])
        edge_colors.append(hex_color)
    
    # Basic graph plot
    ox.plot_graph(G,
                 node_size=5,
                 node_color='black',
                 edge_color=edge_colors,
                 edge_linewidth=2,
                 edge_alpha=0.7,
                 ax=ax,
                 show=False)
    
    # If a path is provided, highlight it
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        ox.plot_graph_route(G, 
                          path,
                          route_color='blue',
                          route_linewidth=4,
                          route_alpha=0.6,
                          ax=ax,
                          show=False)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=norm)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f'Edge Weight (length/votes)', fontsize=12)
    
    # Set title and adjust layout
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    
    return fig, ax


def final_path(G, votes_str, orig_point, dest_point, weight_method='weight'):
    """
    Compute the shortest path between two points in the graph G, using adjusted edge weights.
    
    Parameters:
        G (networkx.MultiDiGraph): the road network graph.
        votes_str (str): JSON string with vote distribution.
        orig_point (tuple): starting point as (lat, lon).
        dest_point (tuple): destination point as (lat, lon).
        weight_method (str): which edge attribute to use ('weight' or "length").
        
    Returns:
        list: list of node IDs representing the shortest path.
    """
    votes_dict = parse_votes_str(votes_str)
    calculate_weight(G, votes_dict)
    # fig, ax = plot_weighted_graph(G, weight_method='weight')
    # plt.show()
    # Note: ox.nearest_nodes expects (longitude, latitude)
    orig_node = ox.nearest_nodes(G, orig_point[1], orig_point[0])
    dest_node = ox.nearest_nodes(G, dest_point[1], dest_point[0])
    shortest_path = nx.shortest_path(G, orig_node, dest_node, weight=weight_method)
    return shortest_path


def process_json(json_filepath, bboxes):
    """
    Process the JSON file containing feature observations.
    
    Parameters:
        json_filepath (str): path to the JSON file.
        bboxes (dict): dictionary of bounding boxes (keys only used to pick random box).
        
    Returns:
        list: list of feature dictionaries with updated 'coordinates'.
    """
    with open(json_filepath, "r") as f:
        json_data = json.load(f)
    
    return json_data


def solve(center_coord, start_coord, dst_coord, commander_intent, json_filepath=DATA_FILE_PATH):
    """
    Generate a road network and bounding box feature map; use GPT to weight areas based on
    commander preferences; then compute and return both a naive and an informed shortest path.
    
    Parameters:
        center_coord (tuple): center coordinate (lat, lon) for the area of interest.
        start_coord (tuple): starting coordinate (lat, lon) for the route.
        dst_coord (tuple): destination coordinate (lat, lon) for the route.
        commander_intent (str): commander's tactical preferences.
        json_filepath (str): path to the JSON file with battlefield features.
        
    Returns:
        dict: containing:
              - "naive_path": list of [lat, lon] for path based solely on length.
              - "informed_path": list of [lat, lon] for path weighted by commander intent.
              - "bbox_info": list of dictionaries with bounding box details and GPT summaries.
    """
    # Step 1: Create the road network.
    G = ox.graph_from_point(center_coord, dist=2000, network_type='all', retain_all=True, simplify=False)
    G = G.to_undirected()
    
    # Step 2: Generate bounding boxes and prepare a structure for features.
    small_boxes_dict = generate_small_boxes(center_coord, box_size=300)
    print("Number of bounding boxes generated:", len(small_boxes_dict))

    bboxes = {bbox: {} for bbox in small_boxes_dict.keys()}
    
    # Step 3: Process external feature data and assign features to boxes.
    json_data = process_json(json_filepath, bboxes)
    assign_features_to_boxes(json_data, bboxes)

    # Only include boxes with some features and convert keys to strings.
    bboxes_filtered = {str(k): v for k, v in bboxes.items() if len(v) > 0}
    
    # Step 4: Prepare GPT prompts.
    SYSTEM_PROMPT = (
        "You are serving a commander of the military; this commander seeks to "
        "understand and optimise his strategy based on a set of his preferences and "
        "given input data. The input data will be JUST a dictionary, where the keys are 4 values in order: the north-most "
        "latitude of the bounding box, the south-most latitude of the box, the east-most longitude of the "
        "bounding box, and the west-most longitude of the bounding box. The bounding boxes "
        "divide a battlefield; bounding boxes with adjacent coordinates in reality represent "
        "areas next to each other. The corresponding values of the keys will be features assigned to each bounding box. "
        "These features will be military observations, such as obstructions or dangers to troops, and so on. "
        "You are given 50000 votes. Distribute all of these votes amongst the bounding boxes by how much their features align with the commander's preferences. Vote exponentially based on how much the features align with the commander's preferences."
        "Format the votes with no commas after three zeros, just the number. One point to note is that all boxes should have at least one vote. "
        "Here are the preferences of the commander, in their words:\n"
        f"\"{commander_intent}\"\n"
        "The output should be specifically and exactly formatted as follows:\n\n"
        "{\n"
        "    '(northmost coord, southmost coord, eastmost coord, westmost coord)': 'votes assigned'\n"
        "}\n\n"
        "The bounding boxes should be arranged in order, east to west, and north to south."
        "Return only this with no additional text, commentary, or rationale. No need to do np.float64() on the keys, just use the numbers as they are."
        "Make sure that all keys and string values in the JSON object are enclosed in double quotes, as required by the JSON standard."
    )
    
    SYSTEM_PROMPT_BBOX = (
        "The user will send a dictionary where the keys are latitude, longitude corners of satellite bounding-boxes,"
        "and the corresponding value is an elaborate description of events going on in the battlefield at the region covered by the bounding box. "
        "Your job is simple: Return only a JSON object, with no additional text or markdown formatting. Take this description and simply summarize the salient points in a sentence or two. "
        "You must output a dictionary where the keys are the exact same as the keys in the dictionary inputted by the user, and the corresponding "
        "value is the description. You must only output the dictionary, no commentary, no markdown formatting of any kind, just the stringified dictionary. "
        "Most importantly, in your response please frame your summaries along the lines of \"in this region there are ...\" where the ellipsis is a placeholder."
    )
    
    # Step 5: Use GPT to distribute votes based on the commander's intent.
    client = OpenAI(api_key=API_KEY)
    response_votes = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(bboxes_filtered)}
        ]
    )
    votes_str = response_votes.choices[0].message.content
    print("GPT Vote Distribution:", votes_str)
    
    # Step 6: Project the graph to EPSG:4326.
    G_proj = ox.project_graph(G, to_crs='EPSG:4326').to_undirected()
    
    # Step 7: Compute both the informed (weighted) and naive (length-based) paths.
    informed_path_nodes = final_path(G_proj, votes_str, start_coord, dst_coord, weight_method='weight')
    naive_path_nodes = final_path(G_proj, votes_str, start_coord, dst_coord, weight_method="length")
    
    # Convert node IDs to (lat, lon) coordinates.
    informed_path = [[G_proj.nodes[n]['y'], G_proj.nodes[n]['x']] for n in informed_path_nodes]
    naive_path = [[G_proj.nodes[n]['y'], G_proj.nodes[n]['x']] for n in naive_path_nodes]
    
    # Step 8: Use GPT to summarize the features in each bounding box.
    response_summary = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BBOX},
            {"role": "user", "content": json.dumps(bboxes_filtered)}
        ]
    )
    summary_response = response_summary.choices[0].message.content
    print("GPT Bounding Box Summaries:", summary_response)
    
    summary_json = json.loads(summary_response)
 

    bbox_infos = []
    votes_dict = parse_votes_str(votes_str)
    print(votes_dict)

    for bbox in small_boxes_dict:
        bbox_str = str(bbox)
        bbox_info = {
            "bounds": bbox,
            "weight": votes_dict.get(bbox, DEFAULT_WEIGHT), # assigning boxes with no features a default
            "summary": summary_json.get(bbox_str, "")
        }
        bbox_infos.append(bbox_info)
    
    return {
        "naive_path": naive_path,
        "informed_path": informed_path,
        "bbox_info": bbox_infos,
    }

if __name__ == '__main__':
    # Starting usage as an example; can be modified via the web UI
    center_coord = (47.1156498, 37.5331467)  # (lat, lon) center of the area
    start_coord = (47.108750, 37.523804)      # (lat, lon) starting point
    dst_coord = (47.121474, 37.542343)        # (lat, lon) destination point
    commander_intent = (
        "I am interested in terrain where there are no enemy operatives nearby, "
        "where the road is unobstructed. I do not wish to pass by enemy radar, "
        "enemy platoons or bases. I do not wish to cross any water."
    )
    
    result = solve(center_coord, start_coord, dst_coord, commander_intent)
    print(json.dumps(result, indent=4))
