from matplotlib import font_manager
import plotnine as pn

from brisk.theme.theme import register_fonts, brisk_theme

class TestBriskTheme():
    def test_register_fonts(self):
        register_fonts()
        font_names = [f.name for f in font_manager.fontManager.ttflist]
        assert "Montserrat" in font_names, "Montserrat font should be registered"

    def test_brisk_theme(self):
        theme = brisk_theme()
        assert isinstance(theme, pn.theme), "brisk_theme should return a valid theme"

        # brisk_theme should register the font
        font_names = [f.name for f in font_manager.fontManager.ttflist]
        assert "Montserrat" in font_names, "Montserrat font should be registered"