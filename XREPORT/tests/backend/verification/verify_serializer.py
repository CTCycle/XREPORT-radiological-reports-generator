import pandas as pd
import json
import sys
import os
import unittest

# Add project root to path
sys.path.append(os.getcwd())

from XREPORT.server.utils.repository.serializer import DataSerializer

class TestDataSerializer(unittest.TestCase):
    def test_parse_json_basic(self):
        """Test basic JSON parsing"""
        print("\nTesting _parse_json basic functionality...")
        
        # Test valid list string
        self.assertEqual(DataSerializer._parse_json('[1, 2, 3]'), [1, 2, 3])
        
        # Test valid dict string
        self.assertEqual(DataSerializer._parse_json('{"a": 1}'), {"a": 1})
        
        # Test already list
        self.assertEqual(DataSerializer._parse_json([1, 2]), [1, 2])
        
        # Test already dict
        self.assertEqual(DataSerializer._parse_json({"b": 2}), {"b": 2})

    def test_parse_json_defaults(self):
        """Test default values handling"""
        print("Testing _parse_json defaults...")
        
        # Test malformed JSON with default (should be {})
        self.assertEqual(DataSerializer._parse_json("{invalid"), {})
        
        # Test malformed JSON with custom default
        self.assertEqual(DataSerializer._parse_json("{invalid", default=[]), [])
        
        # Test None with default
        self.assertEqual(DataSerializer._parse_json(None), {})
        
        # Test None with custom default
        self.assertEqual(DataSerializer._parse_json(None, default=None), None)

    def test_integration_serialization(self):
        """Test the existing integration serialization logic"""
        print("Testing generic serialization integration...")
        serializer = DataSerializer()
        
        # Create test data
        data = {
            'id': [1, 2],
            'tokens': [[1, 2, 3], [4, 5]],  # List column
            'meta': [{'a': 1}, {'b': 2}],   # Dict column
            'text': ['abc', 'def']          # String column
        }
        df = pd.DataFrame(data)
        
        # Run serialization
        serialized = serializer._serialize_json_columns(df)
        
        # Verification serialization happened
        self.assertIsInstance(serialized.iloc[0]['tokens'], str)
        self.assertIsInstance(serialized.iloc[0]['meta'], str)
        self.assertIsInstance(serialized.iloc[0]['text'], str)
        
        # Verify content
        decoded_tokens = DataSerializer._parse_json(serialized.iloc[0]['tokens'])
        self.assertEqual(decoded_tokens, [1, 2, 3])

if __name__ == "__main__":
    with open("verify_output.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner, exit=False)
