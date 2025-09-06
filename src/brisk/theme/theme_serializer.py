"""
JSON serialization with embedded pickle support for plotnine themes.

This module provides a way to serialize complex plotnine theme objects to JSON
by embedding pickled data as base64 strings. This allows for perfect object
serialization while maintaining JSON compatibility for the rest of the data.
"""
import json
import pickle
import base64
import hashlib
from typing import Dict, Any

class PickleJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can embed pickled objects as base64 strings."""
    
    def default(self, obj):
        if hasattr(obj, '__class__') and 'plotnine' in str(type(obj)):
            pickled_data = pickle.dumps(obj)
            b64_data = base64.b64encode(pickled_data).decode('utf-8')

            return {
                '_pickled_object': True,
                '_type': obj.__class__.__name__,
                '_module': obj.__class__.__module__,
                '_data': b64_data,
                '_checksum': hashlib.md5(pickled_data).hexdigest()
            }

        return super().default(obj)


class PickleJSONDecoder:
    """Custom JSON decoder that can restore pickled objects from base64 strings."""
    
    @staticmethod
    def decode_hook(obj):
        """Hook function for json.loads to decode pickled objects."""
        if isinstance(obj, dict) and obj.get('_pickled_object'):
            try:
                b64_data = obj['_data']
                pickled_data = base64.b64decode(b64_data.encode('utf-8'))
                
                expected_checksum = obj.get('_checksum')
                if expected_checksum:
                    actual_checksum = hashlib.md5(pickled_data).hexdigest()
                    if actual_checksum != expected_checksum:
                        raise ValueError("Checksum mismatch - data may be corrupted")
                
                return pickle.loads(pickled_data)
                
            except Exception as e:
                print(f"Warning: Could not deserialize pickled object: {e}")
                return obj
        
        return obj


class ThemePickleJSONSerializer:
    """
    Serialize plotnine themes to JSON with embedded pickle data.
    Combines JSON readability with pickle's perfect object serialization.
    """
    
    def __init__(self):
        self.encoder = PickleJSONEncoder
        self.decoder = PickleJSONDecoder
    
    def theme_to_json(self, theme_obj) -> str:
        """
        Serialize theme object to JSON with embedded pickle data.
        
        Parameters
        ----------
        theme_obj : plotnine theme object
            The theme to serialize
        include_metadata : bool, default True
            Whether to include metadata in JSON
        
        Returns
        -------
        str
            JSON string containing pickled theme
        """
        container = {
            'theme': theme_obj
        }
        
        return json.dumps(container, cls=self.encoder, separators=(',', ':'))
    
    def theme_from_json(self, json_str: str):
        """
        Deserialize theme object from JSON with embedded pickle data.
        
        Parameters
        ----------
        json_str : str
            JSON string containing pickled theme
        
        Returns
        -------
        plotnine theme object
            The deserialized theme
        """
        data = json.loads(json_str, object_hook=self.decoder.decode_hook)
        
        if 'theme' in data:
            return data['theme']
        else:
            return data
    
    def get_theme_info(self, json_str: str) -> Dict[str, Any]:
        """Extract metadata from JSON without fully deserializing the theme."""
        try:
            raw_data = json.loads(json_str)
            
            info = {}
             
            if 'theme' in raw_data and isinstance(raw_data['theme'], dict):
                pickle_info = raw_data['theme']
                if pickle_info.get('_pickled_object'):
                    info['pickled_type'] = pickle_info.get('_type')
                    info['pickled_module'] = pickle_info.get('_module')
                    info['data_size_bytes'] = len(pickle_info.get('_data', '')) * 3 // 4
                    info['has_checksum'] = '_checksum' in pickle_info
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
