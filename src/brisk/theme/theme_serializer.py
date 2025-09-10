"""JSON serialization with embedded pickle support for plotnine themes.

This module provides comprehensive serialization functionality for complex
plotnine theme objects, enabling them to be stored in JSON format while
maintaining perfect object fidelity. It combines JSON readability with
pickle's complete object serialization capabilities.

The module implements a hybrid approach where plotnine theme objects are
pickled and embedded as base64-encoded strings within JSON structures.
This allows for perfect reconstruction of complex theme objects while
maintaining JSON compatibility for configuration storage and transfer.

Classes
-------
PickleJSONEncoder
    Custom JSON encoder that embeds pickled objects as base64 strings
PickleJSONDecoder
    Custom JSON decoder that restores pickled objects from base64 strings
ThemePickleJSONSerializer
    Main serializer class for plotnine theme objects

Examples
--------
>>> from brisk.theme.theme_serializer import ThemePickleJSONSerializer
>>> import plotnine as pn
>>> 
>>> # Create a theme
>>> theme = pn.theme_minimal() + pn.theme(text=pn.element_text(size=14))
>>> 
>>> # Serialize to JSON
>>> serializer = ThemePickleJSONSerializer()
>>> json_str = serializer.theme_to_json(theme)
>>> 
>>> # Deserialize from JSON
>>> restored_theme = serializer.theme_from_json(json_str)
>>> 
>>> # Get theme metadata
>>> info = serializer.get_theme_info(json_str)
>>> print(info["pickled_type"])  # "theme"
"""

import json
import pickle
import base64
import hashlib
from typing import Dict, Any

class PickleJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can embed pickled objects as base64 strings.
    
    This encoder extends the standard JSON encoder to handle plotnine theme
    objects by pickling them and embedding the pickled data as base64-encoded
    strings within the JSON structure. This allows for perfect serialization
    of complex theme objects while maintaining JSON compatibility.
    
    The encoder automatically detects plotnine objects and converts them to
    a special JSON structure containing the pickled data, metadata, and
    integrity checksum.
    
    Notes
    -----
    The encoder creates a special JSON structure for plotnine objects:
    - _pickled_object: Boolean flag indicating this is a pickled object
    - _type: Class name of the original object
    - _module: Module name where the class is defined
    - _data: Base64-encoded pickled data
    - _checksum: MD5 checksum for data integrity verification
    
    Examples
    --------
    >>> import plotnine as pn
    >>> from brisk.theme.theme_serializer import PickleJSONEncoder
    >>> import json
    >>> 
    >>> theme = pn.theme_minimal()
    >>> encoder = PickleJSONEncoder()
    >>> json_str = json.dumps(theme, cls=encoder)
    """

    def default(self, o: Any) -> Any:
        """Encode plotnine objects as pickled base64 strings.
        
        This method overrides the default JSON encoding behavior to handle
        plotnine theme objects specially. It detects plotnine objects and
        converts them to a JSON-compatible structure containing pickled data.
        
        Parameters
        ----------
        o : Any
            The object to encode
            
        Returns
        -------
        Any
            JSON-serializable representation of the object, or calls
            super().default(o) for non-plotnine objects
            
        Notes
        -----
        The method checks if the object is a plotnine object by looking for
        "plotnine" in the type string. If it is, it pickles the object,
        encodes it as base64, and wraps it in a special JSON structure.
        """
        if hasattr(o, "__class__") and "plotnine" in str(type(o)):
            pickled_data = pickle.dumps(o)
            b64_data = base64.b64encode(pickled_data).decode("utf-8")

            return {
                "_pickled_object": True,
                "_type": o.__class__.__name__,
                "_module": o.__class__.__module__,
                "_data": b64_data,
                "_checksum": hashlib.md5(pickled_data).hexdigest()
            }

        return super().default(o)


class PickleJSONDecoder:
    """Custom JSON decoder that can restore pickled objects from base64 strings.
    
    This decoder provides functionality to restore plotnine theme objects
    from JSON structures that contain embedded pickled data. It works as
    a hook function for json.loads to automatically detect and deserialize
    pickled objects during JSON parsing.
    
    The decoder includes integrity checking through MD5 checksums to ensure
    that the pickled data has not been corrupted during storage or transfer.
    
    Notes
    -----
    The decoder looks for objects with the "_pickled_object" flag and
    attempts to deserialize them. If deserialization fails or checksums
    don't match, it returns the original object and prints a warning.
    
    Examples
    --------
    >>> from brisk.theme.theme_serializer import PickleJSONDecoder
    >>> import json
    >>> 
    >>> # JSON string containing pickled theme
    >>> json_str = '{"theme": {"_pickled_object": true, ...}}'
    >>> data = json.loads(json_str, object_hook=PickleJSONDecoder.decode_hook)
    >>> theme = data["theme"]  # Restored plotnine theme object
    """

    @staticmethod
    def decode_hook(obj: Any) -> Any:
        """Hook function for json.loads to decode pickled objects.
        
        This static method serves as a hook function for json.loads to
        automatically detect and deserialize pickled objects embedded in
        JSON structures. It includes integrity checking and error handling.
        
        Parameters
        ----------
        obj : Any
            The object being processed during JSON deserialization
            
        Returns
        -------
        Any
            The deserialized object if it was pickled, or the original
            object if it wasn't a pickled object
            
        Notes
        -----
        The method checks for the "_pickled_object" flag and attempts to
        deserialize the object. It verifies data integrity using MD5
        checksums and handles errors gracefully by returning the original
        object and printing a warning.
        """
        if isinstance(obj, dict) and obj.get("_pickled_object"):
            try:
                b64_data = obj["_data"]
                pickled_data = base64.b64decode(b64_data.encode("utf-8"))

                expected_checksum = obj.get("_checksum")
                if expected_checksum:
                    actual_checksum = hashlib.md5(pickled_data).hexdigest()
                    if actual_checksum != expected_checksum:
                        raise ValueError(
                            "Checksum mismatch - data may be corrupted"
                        )

                return pickle.loads(pickled_data)

            except (pickle.UnpicklingError, ValueError) as e:
                print(f"Warning: Could not deserialize pickled object: {e}")
                return obj

        return obj


class ThemePickleJSONSerializer:
    """Serialize plotnine themes to JSON with embedded pickle data.
    
    This class provides a high-level interface for serializing and
    deserializing plotnine theme objects to/from JSON format. It combines
    JSON readability with pickle's perfect object serialization capabilities,
    allowing complex theme objects to be stored and transferred as JSON
    while maintaining complete fidelity.
    
    The serializer uses the PickleJSONEncoder and PickleJSONDecoder classes
    internally to handle the conversion between plotnine objects and JSON
    structures with embedded pickled data.
    
    Attributes
    ----------
    encoder : PickleJSONEncoder
        The JSON encoder class for serializing plotnine objects
    decoder : PickleJSONDecoder
        The JSON decoder class for deserializing pickled objects
        
    Notes
    -----
    This serializer is specifically designed for plotnine theme objects
    but can handle any plotnine object that contains "plotnine" in its
    type string.
    
    Examples
    --------
    >>> from brisk.theme.theme_serializer import ThemePickleJSONSerializer
    >>> import plotnine as pn
    >>> 
    >>> # Create a theme
    >>> theme = pn.theme_minimal() + pn.theme(text=pn.element_text(size=14))
    >>> 
    >>> # Serialize to JSON
    >>> serializer = ThemePickleJSONSerializer()
    >>> json_str = serializer.theme_to_json(theme)
    >>> 
    >>> # Deserialize from JSON
    >>> restored_theme = serializer.theme_from_json(json_str)
    >>> 
    >>> # Get theme metadata
    >>> info = serializer.get_theme_info(json_str)
    """

    def __init__(self) -> None:
        """Initialize the theme serializer.
        
        This constructor sets up the serializer with the appropriate
        encoder and decoder classes for handling plotnine objects.
        """
        self.encoder = PickleJSONEncoder
        self.decoder = PickleJSONDecoder

    def theme_to_json(self, theme_obj: Any) -> str:
        """Serialize theme object to JSON with embedded pickle data.
        
        This method converts a plotnine theme object into a JSON string
        that contains the pickled theme data. The theme is wrapped in a
        container object and serialized using the custom encoder.
        
        Parameters
        ----------
        theme_obj : Any
            The plotnine theme object to serialize
            
        Returns
        -------
        str
            JSON string containing the pickled theme object
            
        Notes
        -----
        The method wraps the theme object in a container with the key
        "theme" and uses the custom PickleJSONEncoder to handle the
        serialization. The resulting JSON is compact with no extra spaces.
        
        Examples
        --------
        >>> from brisk.theme.theme_serializer import ThemePickleJSONSerializer
        >>> import plotnine as pn
        >>> 
        >>> theme = pn.theme_minimal()
        >>> serializer = ThemePickleJSONSerializer()
        >>> json_str = serializer.theme_to_json(theme)
        >>> print(json_str[:100])  # Shows start of JSON string
        """
        container = {
            "theme": theme_obj
        }

        return json.dumps(container, cls=self.encoder, separators=(",", ":"))

    def theme_from_json(self, json_str: str) -> Any:
        """Deserialize theme object from JSON with embedded pickle data.
        
        This method converts a JSON string containing pickled theme data
        back into a plotnine theme object. It uses the custom decoder
        to handle the deserialization of pickled objects.
        
        Parameters
        ----------
        json_str : str
            JSON string containing the pickled theme object
            
        Returns
        -------
        Any
            The deserialized plotnine theme object
            
        Notes
        -----
        The method uses json.loads with the custom decoder hook to
        automatically deserialize any pickled objects in the JSON.
        If the JSON contains a "theme" key, it returns that value;
        otherwise, it returns the entire deserialized data.
        
        Examples
        --------
        >>> from brisk.theme.theme_serializer import ThemePickleJSONSerializer
        >>> 
        >>> # JSON string containing pickled theme
        >>> json_str = '{"theme": {"_pickled_object": true, ...}}'
        >>> serializer = ThemePickleJSONSerializer()
        >>> theme = serializer.theme_from_json(json_str)
        >>> # theme is now a plotnine theme object
        """
        data = json.loads(json_str, object_hook=self.decoder.decode_hook)

        if "theme" in data:
            return data["theme"]
        else:
            return data

    def get_theme_info(self, json_str: str) -> Dict[str, Any]:
        """Extract metadata from JSON without fully deserializing the theme.
        
        This method extracts metadata about a pickled theme object from
        a JSON string without actually deserializing the theme. This is
        useful for inspecting theme properties without the overhead of
        full deserialization.
        
        Parameters
        ----------
        json_str : str
            JSON string containing the pickled theme object
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing metadata about the pickled theme:
            - pickled_type: Class name of the pickled object
            - pickled_module: Module where the class is defined
            - data_size_bytes: Approximate size of the pickled data
            - has_checksum: Whether the data includes a checksum
            - error: Error message if parsing failed
            
        Notes
        -----
        The method calculates the approximate size of the pickled data
        by multiplying the base64 string length by 3/4, which gives
        a rough estimate of the original binary data size.
        
        Examples
        --------
        >>> from brisk.theme.theme_serializer import ThemePickleJSONSerializer
        >>> import plotnine as pn
        >>> 
        >>> theme = pn.theme_minimal()
        >>> serializer = ThemePickleJSONSerializer()
        >>> json_str = serializer.theme_to_json(theme)
        >>> info = serializer.get_theme_info(json_str)
        >>> print(f"Type: {info['pickled_type']}")
        >>> print(f"Size: {info['data_size_bytes']} bytes")
        """
        try:
            raw_data = json.loads(json_str)

            info = {}

            if "theme" in raw_data and isinstance(raw_data["theme"], dict):
                pickle_info = raw_data["theme"]
                if pickle_info.get("_pickled_object"):
                    info["pickled_type"] = pickle_info.get("_type")
                    info["pickled_module"] = pickle_info.get("_module")
                    info["data_size_bytes"] = (
                        len(pickle_info.get("_data", "")) * 3 // 4
                    )
                    info["has_checksum"] = "_checksum" in pickle_info

            return info

        except (json.JSONDecodeError, TypeError) as e:
            return {"error": str(e)}
