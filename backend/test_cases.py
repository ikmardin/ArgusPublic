import unittest
from unittest.mock import patch, MagicMock
import json
import networkx as nx
from shapely.geometry import Polygon, LineString, Point
import geopandas as gpd
from testing import (
    generate_small_boxes,
    in_bbox,
    assign_features_to_boxes,
    parse_votes_str,
    calculate_weight,
    process_json,
    solve
)

class TestRoutePlanning(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.center_point = (47.1156498, 37.5331467)
        # Define multiple bounding boxes
        self.test_bbox1 = (47.12, 47.11, 37.54, 37.53)
        self.test_bbox2 = (47.11, 47.10, 37.53, 37.52)
        self.test_bbox3 = (47.13, 47.12, 37.55, 37.54)
        
        # Features in different locations
        self.test_features = [
            {
                'coordinates': '37.535, 47.115',  # For bbox1
                'description': 'Enemy patrol spotted'
            },
            {
                'coordinates': '37.525, 47.105',  # For bbox2
                'description': 'Road obstruction'
            },
            {
                'coordinates': '37.545, 47.125',  # For bbox3
                'description': 'Supply depot'
            }
        ]
        
    def test_generate_small_boxes(self):
        """Test generation of bounding boxes"""
        boxes = generate_small_boxes(self.center_point, box_size=100)
        self.assertIsInstance(boxes, dict)
        # Check if boxes are Polygons
        for box in boxes.values():
            self.assertIsInstance(box, Polygon)
        # Number of boxes may vary based on area and box_size
        # Instead of exact count, verify reasonable range
        self.assertGreater(len(boxes), 100)  # Should have sufficient coverage
        self.assertLess(len(boxes), 2000)    # Shouldn't be excessive
        
        # Verify the bounding box coordinates are within expected range
        for coords, box in boxes.items():
            north, south, east, west = coords
            self.assertGreater(north, south)
            self.assertGreater(east, west)
            self.assertGreater(north, 47.0)
            self.assertLess(north, 48.0)
            self.assertGreater(east, 37.0)
            self.assertLess(east, 38.0)

    def test_in_bbox(self):
        """Test point in bounding box detection"""
        # Point inside bbox1
        self.assertTrue(in_bbox(self.test_bbox1, 47.115, 37.535))
        # Point inside bbox2
        self.assertTrue(in_bbox(self.test_bbox2, 47.105, 37.525))
        # Point inside bbox3
        self.assertTrue(in_bbox(self.test_bbox3, 47.125, 37.545))
        # Point outside all boxes
        self.assertFalse(in_bbox(self.test_bbox1, 47.13, 37.55))
        self.assertFalse(in_bbox(self.test_bbox2, 47.13, 37.55))
        self.assertFalse(in_bbox(self.test_bbox3, 47.09, 37.51))

    def test_assign_features_to_boxes(self):
        """Test feature assignment to boxes"""
        # Initialize boxes dictionary with multiple bounding boxes
        boxes = {
            self.test_bbox1: {},
            self.test_bbox2: {},
            self.test_bbox3: {}
        }
        
        # Assign all features
        assign_features_to_boxes(self.test_features, boxes)
        
        # Check each box has correct features
        # Box 1 should have the enemy patrol
        self.assertIn('features', boxes[self.test_bbox1])
        found_patrol = False
        for feature in boxes[self.test_bbox1].get('features', []):
            if feature['description'] == 'Enemy patrol spotted':
                found_patrol = True
        self.assertTrue(found_patrol, "Enemy patrol should be in bbox1")
        
        # Box 2 should have the road obstruction
        self.assertIn('features', boxes[self.test_bbox2])
        found_obstruction = False
        for feature in boxes[self.test_bbox2].get('features', []):
            if feature['description'] == 'Road obstruction':
                found_obstruction = True
        self.assertTrue(found_obstruction, "Road obstruction should be in bbox2")
        
        # Box 3 should have the supply depot
        self.assertIn('features', boxes[self.test_bbox3])
        found_depot = False
        for feature in boxes[self.test_bbox3].get('features', []):
            if feature['description'] == 'Supply depot':
                found_depot = True
        self.assertTrue(found_depot, "Supply depot should be in bbox3")
        
        # Test feature outside all boxes
        outside_feature = [{
            'coordinates': '37.7, 47.3',
            'description': 'Outside feature'
        }]
        boxes_empty = {
            self.test_bbox1: {},
            self.test_bbox2: {},
            self.test_bbox3: {}
        }
        assign_features_to_boxes(outside_feature, boxes_empty)
        for bbox in boxes_empty.values():
            self.assertNotIn('features', bbox)

    def test_parse_votes_str(self):
        """Test parsing of votes string"""
        # Test votes distributed across multiple boxes
        votes_str = '''{
            "(47.12, 47.11, 37.54, 37.53)": "20000",
            "(47.11, 47.10, 37.53, 37.52)": "15000",
            "(47.13, 47.12, 37.55, 37.54)": "15000"
        }'''
        result = parse_votes_str(votes_str)
        self.assertIsInstance(result, dict)
        self.assertEqual(result[(47.12, 47.11, 37.54, 37.53)], 20000)
        self.assertEqual(result[(47.11, 47.10, 37.53, 37.52)], 15000)
        self.assertEqual(result[(47.13, 47.12, 37.55, 37.54)], 15000)
        
        # Test invalid JSON
        invalid_votes = 'invalid json'
        self.assertEqual(parse_votes_str(invalid_votes), {})
        
        # Test malformed coordinates
        malformed_votes = '{"(invalid, coords)": "1000"}'
        self.assertEqual(parse_votes_str(malformed_votes), {})
        
        # Test empty string
        self.assertEqual(parse_votes_str(""), {})

    def test_calculate_weight(self):
        """Test weight calculation for graph edges"""
        G = nx.MultiDiGraph()
        # Add test nodes and edges
        G.add_edge(0, 1, geometry=LineString([(37.53, 47.11), (37.54, 47.12)]), length=100)
        
        # Create votes dictionary with distribution across multiple boxes
        votes_dict = {
            self.test_bbox1: 20000,
            self.test_bbox2: 15000,
            self.test_bbox3: 15000
        }
        
        calculate_weight(G, votes_dict)
        self.assertIn('weight', G.edges[0, 1, 0])
        # Weight should be length divided by minimum votes along the edge
        expected_weight = 100 / 15000  # Using minimum vote value
        self.assertAlmostEqual(G.edges[0, 1, 0]['weight'], expected_weight, places=5)

    @patch('openai.OpenAI')
    def test_solve(self, mock_openai):
        """Test the main solve function"""
        # Mock OpenAI responses
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content='{"(47.12, 47.11, 37.54, 37.53)": "1000"}'))
        ]
        mock_openai.return_value.chat.completions.create.return_value = mock_completion

        start_coord = (47.108750, 37.523804)
        dst_coord = (47.121474, 37.542343)
        commander_intent = "Avoid enemy patrols and water"

        result = solve(
            self.center_point,
            start_coord,
            dst_coord,
            commander_intent,
            json_filepath="test_data.json"
        )

        # Check result structure
        self.assertIn('naive_path', result)
        self.assertIn('informed_path', result)
        self.assertIn('bbox_info', result)
        
        # Check paths are lists of coordinates
        self.assertIsInstance(result['naive_path'], list)
        self.assertIsInstance(result['informed_path'], list)
        
        # Check bbox_info structure
        self.assertIsInstance(result['bbox_info'], list)
        for bbox in result['bbox_info']:
            self.assertIn('bounds', bbox)
            self.assertIn('weight', bbox)
            self.assertIn('summary', bbox)

    def test_process_json(self):
        """Test JSON processing"""
        # Create a temporary test JSON file
        test_data = [
            {
                'coordinates': '37.535, 47.115',
                'description': 'Feature in box 1'
            },
            {
                'coordinates': '37.525, 47.105',
                'description': 'Feature in box 2'
            },
            {
                'coordinates': '37.545, 47.125',
                'description': 'Feature in box 3'
            }
        ]
        with open('test_data.json', 'w') as f:
            json.dump(test_data, f)

        bboxes = {
            self.test_bbox1: {},
            self.test_bbox2: {},
            self.test_bbox3: {}
        }
        result = process_json('test_data.json', bboxes)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['description'], 'Feature in box 1')
        self.assertEqual(result[1]['description'], 'Feature in box 2')
        self.assertEqual(result[2]['description'], 'Feature in box 3')

if __name__ == '__main__':
    unittest.main()