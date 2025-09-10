"""Plot settings and theme management for Brisk visualization.

This module provides comprehensive plot customization functionality for the
Brisk package, including file I/O settings, theme management, and color
configuration. It serves as a centralized way to control all aspects of
plot generation and styling.

The PlotSettings class allows users to customize:
- File output settings (file_format, size, DPI, transparency)
- Plot styling using plotnine theme objects
- Color schemes and palettes
- Theme inheritance and override behavior

Classes
-------
PlotSettings
    Main class for controlling plot styling and file I/O settings

Examples
--------
>>> from brisk.theme.plot_settings import PlotSettings
>>> import plotnine as pn
>>> 
>>> # Basic plot settings
>>> plot_settings = PlotSettings(
...     file_format="svg",
...     width=12,
...     height=8,
...     dpi=300
... )
>>> 
>>> # Custom theme
>>> custom_theme = pn.theme_minimal() + pn.theme(text=pn.element_text(size=14))
>>> plot_settings = PlotSettings(
...     theme=custom_theme,
...     override=True,
...     primary_color="#FF6B6B"
... )
>>> 
>>> # Get settings for use
>>> io_settings = plot_settings.get_io_settings()
>>> theme = plot_settings.get_theme()
"""

from typing import Optional, Dict, Any

import plotnine as pn

from brisk.theme.theme import brisk_theme, register_fonts
from brisk.theme import theme_serializer

class PlotSettings:
    """Control the styling and file I/O settings for plots.

    This class provides a centralized way to control all aspects of plot
    generation and styling in the Brisk package. It manages both file I/O
    settings (file_format, size, DPI, etc.) and plot styling using plotnine
    theme objects.

    The plot styling settings apply to built-in plots automatically and to
    custom plotnine plots via the get_theme() method. File I/O settings
    are used by the IOService when saving plots.

    Attributes
    ----------
    VALID_FORMATS : set
        Set of valid file formats for plot output
    file_io_settings : Dict[str, Any]
        Dictionary containing file I/O settings (file_format, width, height,
        dpi, transparent)
    primary_color : str
        Primary color for plots (default: "#1175D5")
    secondary_color : str
        Secondary color for plots (default: "#00A878")
    accent_color : str
        Accent color for plots (default: "#DE6B48")
    theme : pn.theme
        The plotnine theme object for plot styling

    Notes
    -----
    The class supports theme inheritance where custom themes can either
    override the default brisk theme completely or extend it by combining
    with the default theme.

    Examples
    --------
    >>> from brisk.theme.plot_settings import PlotSettings
    >>> import plotnine as pn
    >>> 
    >>> # Basic settings
    >>> plot_settings = PlotSettings(file_format="svg", width=12, height=8)
    >>> 
    >>> # Custom theme with override
    >>> custom_theme = pn.theme_minimal()
    >>> plot_settings = PlotSettings(theme=custom_theme, override=True)
    >>> 
    >>> # Custom colors
    >>> plot_settings = PlotSettings(
    ...     primary_color="#FF6B6B",
    ...     secondary_color="#4ECDC4",
    ...     accent_color="#45B7D1"
    ... )
    """
    VALID_FORMATS = {"png", "jpg", "jpeg", "svg", "pdf"}

    def __init__(
        self,
        theme: Optional[pn.theme] = None,
        override: bool = False,
        file_format: str = "png",
        width: int = 10,
        height: int = 8,
        dpi: int = 300,
        transparent: bool = False,
        primary_color: Optional[str] = None,
        secondary_color: Optional[str] = None,
        accent_color: Optional[str] = None
    ) -> None:
        """Initialize PlotSettings with custom parameters.
        
        This constructor sets up the plot settings with the specified theme,
        file I/O settings, and color scheme. It validates the theme object
        and file file_format, and sets up the appropriate theme inheritance.
        
        Parameters
        ----------
        theme : Optional[pn.theme], default=None
            A plotnine theme object. If None, uses the default brisk theme.
            Can be any plotnine theme such as:
            - Built-in themes: theme_minimal(), theme_classic(), etc.
            - Custom themes: your_custom_theme()
            - Combined themes: theme_minimal() + theme(text=element_text(...))
        override : bool, default=False
            If True, override the default theme with the provided theme.
            If False, extend the default theme with the provided theme.
        file_format : str, default="png"
            File file_format for plot output. Must be one of: png, jpg, jpeg,
            svg, pdf
        width : int, default=10
            Figure width in inches
        height : int, default=8
            Figure height in inches
        dpi : int, default=300
            Resolution in dots per inch
        transparent : bool, default=False
            Whether to use transparent background
        primary_color : Optional[str], default=None
            Primary color for plots (default: "#1175D5")
        secondary_color : Optional[str], default=None
            Secondary color for plots (default: "#00A878")
        accent_color : Optional[str], default=None
            Accent color for plots (default: "#DE6B48")
            
        Raises
        ------
        ValueError
            If the file file_format is not in VALID_FORMATS
        TypeError
            If the theme is not a plotnine theme object
            
        Notes
        -----
        The method automatically registers fonts and validates the theme
        object. Color values use default colors if not provided.
        """
        self.file_io_settings = {}
        self._update_io_settings(
            file_format, width, height, dpi, transparent
        )

        self.primary_color = primary_color or "#1175D5"
        self.secondary_color = secondary_color or "#00A878"
        self.accent_color = accent_color or "#DE6B48"
        register_fonts()
        if theme is not None:
            self._validate_theme(theme)
            if override:
                self.theme = theme
            else:
                self.theme = brisk_theme() + theme
        else:
            self.theme = brisk_theme()

    def _update_io_settings(
        self,
        file_format: str,
        width: int,
        height: int,
        dpi: int,
        transparent: bool
    ) -> None:
        """Update file I/O settings and validate file_format.
        
        This private method updates the file I/O settings dictionary with
        the provided parameters and validates that the file file_format is
        supported.
        
        Parameters
        ----------
        file_format : str
            The file file_format for plot output
        width : int
            Figure width in inches
        height : int
            Figure height in inches
        dpi : int
            Resolution in dots per inch
        transparent : bool
            Whether to use transparent background
            
        Raises
        ------
        ValueError
            If the file file_format is not in VALID_FORMATS
        """
        self.file_io_settings["file_format"] = file_format
        self.file_io_settings["width"] = width
        self.file_io_settings["height"] = height
        self.file_io_settings["dpi"] = dpi
        self.file_io_settings["transparent"] = transparent
        if self.file_io_settings["file_format"] not in self.VALID_FORMATS:
            raise ValueError(
                "Invalid file file_format: "
                f"{self.file_io_settings['file_format']}. "
                f"Valid formats are: {', '.join(self.VALID_FORMATS)}"
            )

    def _validate_theme(self, theme: pn.theme) -> None:
        """Validate that the provided theme is a plotnine theme object.
        
        This private method validates that the provided theme object is
        a valid plotnine theme instance.
        
        Parameters
        ----------
        theme : pn.theme
            The theme object to validate
            
        Raises
        ------
        TypeError
            If the theme is not a plotnine theme object
        """
        if not isinstance(theme, pn.theme):
            raise TypeError(
                f"theme must be a plotnine theme object, got {type(theme)}"
            )

    def get_theme(self) -> pn.theme:
        """Get the current theme object.
        
        This method returns the current plotnine theme object that is
        being used for plot styling.
        
        Returns
        -------
        pn.theme
            The current plotnine theme object
            
        Examples
        --------
        >>> plot_settings = PlotSettings()
        >>> theme = plot_settings.get_theme()
        >>> # Use theme with plotnine plots
        >>> p = pn.ggplot(data) + pn.geom_point() + theme
        """
        return self.theme

    def get_io_settings(self) -> Dict[str, Any]:
        """Get the current file I/O settings.
        
        This method returns a copy of the current file I/O settings
        dictionary containing file_format, width, height, DPI, and transparency
        settings.
        
        Returns
        -------
        Dict[str, Any]
            A copy of the file I/O settings dictionary containing:
            - file_format: File file_format (str)
            - width: Figure width in inches (int)
            - height: Figure height in inches (int)
            - dpi: Resolution in dots per inch (int)
            - transparent: Whether to use transparent background (bool)
            
        Examples
        --------
        >>> plot_settings = PlotSettings(file_format="svg", width=12, height=8)
        >>> io_settings = plot_settings.get_io_settings()
        >>> print(io_settings["file_format"])  # "svg"
        >>> print(io_settings["width"])   # 12
        """
        return self.file_io_settings.copy()

    def get_colors(self) -> Dict[str, str]:
        """Get the current color settings as a dictionary.
        
        This method returns a dictionary containing all the current
        color settings for the plot theme.
        
        Returns
        -------
        Dict[str, str]
            A dictionary containing:
            - primary_color: Primary color for plots (str)
            - secondary_color: Secondary color for plots (str)
            - accent_color: Accent color for plots (str)
            
        Examples
        --------
        >>> plot_settings = PlotSettings(
        ...     primary_color="#FF6B6B",
        ...     secondary_color="#4ECDC4",
        ...     accent_color="#45B7D1"
        ... )
        >>> colors = plot_settings.get_colors()
        >>> print(colors["primary_color"])  # "#FF6B6B"
        """
        return {
            "primary_color": self.primary_color,
            "secondary_color": self.secondary_color,
            "accent_color": self.accent_color
        }

    def export_params(self) -> Dict[str, Any]:
        """Export PlotSettings to a JSON-serializable dictionary.
        
        This method exports all PlotSettings parameters to a dictionary
        that can be serialized to JSON. The theme object is serialized
        using a custom serializer to make it JSON-compatible.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all PlotSettings parameters:
            - file_io_settings: File I/O settings dictionary
            - colors: Color settings dictionary
            - theme_json: Serialized theme object as JSON-compatible string
            
        Notes
        -----
        The theme object is serialized using ThemePickleJSONSerializer
        to convert the plotnine theme object into a JSON-compatible
        file_format that can be reconstructed later.
        
        Examples
        --------
        >>> plot_settings = PlotSettings(file_format="svg", width=12, height=8)
        >>> params = plot_settings.export_params()
        >>> # Save to JSON file
        >>> import json
        >>> with open("plot_settings.json", "w") as f:
        ...     json.dump(params, f)
        """
        serializer = theme_serializer.ThemePickleJSONSerializer()
        return {
            "file_io_settings": self.get_io_settings(),
            "colors": self.get_colors(),
            "theme_json": serializer.theme_to_json(self.theme),
        }
