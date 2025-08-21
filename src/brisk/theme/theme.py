"""Custom theme for brisk package plots."""
import os

from matplotlib import font_manager
import plotnine as pn

def register_fonts():
    theme_dir = os.path.dirname(os.path.abspath(__file__))
    font_paths = {
        "montserrat": os.path.join(theme_dir, "Montserrat-Regular.ttf"),
    }

    for name, font_path in font_paths.items():
        try:
            font_manager.fontManager.addfont(font_path)
        except (ValueError, TypeError) as e:
            print(f"Failed to register font {name}: {e}")


def brisk_theme():
    """Create a custom theme for brisk package plots.

    Returns:
        theme: A plotnine theme object with custom styling
    """
    register_fonts()
    return pn.theme(
        complete=False,

        # Plot Background
        plot_background=pn.element_rect(fill="white"),
        panel_background=pn.element_rect(fill="#FAF9F6"),

        # Grid
        panel_grid_major=pn.element_line(color="#D3D3D3", size=0.5),
        panel_grid_minor=pn.element_line(color="#FAF9F6", size=0.25),

        # Text elements
        text=pn.element_text(size=16, family="montserrat"),
        axis_text=pn.element_text(size=10),
        axis_title=pn.element_text(size=14),
        title=pn.element_text(size=20),

        # Border
        panel_border=pn.element_rect(color="black", fill=None),
    )
