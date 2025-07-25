from enum import Enum


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
