from enum import Enum

__all__ = ['StrOptions', 'Cmaps', 'SeabornPalettes']

class StrOptions(str, Enum):
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Cmaps(StrOptions):
    """
    Matplotlib colormap options.

    This class provides a set of predefined colormap names with their
    descriptions for use in matplotlib plots.

    Available colormaps
    -------------------
    - ``viridis`` : Purple -> Yellow (default, perceptually uniform and colorblind-friendly)
    - ``plasma`` : Purple -> Pink -> Yellow (perceptually uniform)
    - ``inferno`` : Black -> Red -> Yellow (perceptually uniform)
    - ``magma`` : Black -> Purple -> White (perceptually uniform)
    - ``cividis`` : Colorblind-friendly version of viridis
    - ``Greys`` : White to black grayscale
    - ``Blues`` : White to blue sequential colormap
    - ``Reds`` : White to red sequential colormap
    - ``RdYlBu`` : Red -> Yellow -> Blue (diverging)
    - ``RdYlGn`` : Red -> Yellow -> Green (diverging)

    Examples
    --------
        >>> from pyfmto.utilities.stroptions import Cmaps
        >>> print(Cmaps.viridis)
        viridis
        >>> plot_func_2d(func, lb, ub, dim, cmap=Cmaps.viridis)
    """

    # Sequential colormaps (perceptually uniform)
    viridis = "Purple -> Yellow (default, perceptually uniform and colorblind-friendly)"
    plasma = "Purple -> Pink -> Yellow (perceptually uniform)"
    inferno = "Black -> Red -> Yellow (perceptually uniform)"
    magma = "Black -> Purple -> White (perceptually uniform)"
    cividis = "Colorblind-friendly version of viridis"

    # Sequential colormaps (single hue)
    Greys = "White to black grayscale"
    Blues = "White to blue sequential colormap"
    Reds = "White to red sequential colormap"

    # Diverging colormaps
    RdYlBu = "Red -> Yellow -> Blue (diverging)"
    RdYlGn = "Red -> Yellow -> Green (diverging)"


class SeabornPalettes(StrOptions):
    """
    Seaborn palette options.

    This class provides a set of predefined palette names with their
    descriptions for use in seaborn plots.

    Available palettes
    ------------------
    - ``deep`` : Deep color palette with 10 colors
    - ``muted`` : Muted color palette with 10 colors
    - ``bright`` : Bright color palette with 10 colors
    - ``pastel`` : Pastel color palette with 10 colors
    - ``dark`` : Dark color palette with 10 colors
    - ``colorblind`` : Colorblind-friendly palette with 10 colors
    - ``husl`` : HUSL color space palette
    - ``Set1`` : Brewer Set1 palette
    - ``Set2`` : Brewer Set2 palette
    - ``Set3`` : Brewer Set3 palette
    - ``Paired`` : Brewer Paired palette
    - ``viridis`` : Viridis sequential colormap
    - ``plasma`` : Plasma sequential colormap
    - ``inferno`` : Inferno sequential colormap
    - ``magma`` : Magma sequential colormap

    Examples
    --------
        >>> from pyfmto.utilities.stroptions import SeabornPalettes
        >>> print(SeabornPalettes.deep)
        deep
        >>> sns.scatterplot(data=df, x='x', y='y', hue='category', palette=SeabornPalettes.deep)
    """

    # Seaborn default palettes
    deep = "Deep color palette with 10 colors"
    muted = "Muted color palette with 10 colors"
    bright = "Bright color palette with 10 colors"
    pastel = "Pastel color palette with 10 colors"
    dark = "Dark color palette with 10 colors"
    colorblind = "Colorblind-friendly palette with 10 colors"

    # Other categorical palettes
    husl = "HUSL color space palette"
    Set1 = "Brewer Set1 palette"
    Set2 = "Brewer Set2 palette"
    Set3 = "Brewer Set3 palette"
    Paired = "Brewer Paired palette"

    # Sequential palettes (from matplotlib colormaps)
    viridis = "Viridis sequential colormap"
    plasma = "Plasma sequential colormap"
    inferno = "Inferno sequential colormap"
    magma = "Magma sequential colormap"