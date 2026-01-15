from enum import Enum

__all__ = [
    'Cmaps',
    'SeabornPalettes',
    'SeabornStyles',
    'StrColors',
    'StrOptions',
]


class StrOptions(str, Enum):
    def __str__(self):
        return self.name


class Cmaps(StrOptions):
    """
    Matplotlib colormap options.

    This class provides a set of predefined colormap names with their
    descriptions for use in matplotlib plots.

    Uniform
    -------
    - ``viridis`` : Purple -> Yellow (colorblind-friendly)"
    - ``plasma``  : Purple -> Pink -> Yellow"
    - ``inferno`` : Black -> Red -> Yellow"
    - ``magma``   : Black -> Purple -> White"
    - ``cividis`` : Colorblind-friendly version of viridis"

    Sequential
    ----------
    - ``Greys``   : White to black grayscale"
    - ``Purples`` : White to purple sequential colormap"
    - ``Blues``   : White to blue sequential colormap"
    - ``Greens``  : White to green sequential colormap"
    - ``Oranges`` : White to orange sequential colormap"
    - ``Reds``    : White to red sequential colormap"
    - ``YlOrBr``  : Yellow -> Orange -> Brown sequential"
    - ``YlOrRd``  : Yellow -> Orange -> Red sequential"
    - ``OrRd``    : Orange -> Red sequential"
    - ``PuRd``    : Purple -> Red sequential"
    - ``RdPu``    : Red -> Purple sequential"
    - ``BuPu``    : Blue -> Purple sequential"
    - ``GnBu``    : Green -> Blue sequential"
    - ``PuBu``    : Purple -> Blue sequential"
    - ``YlGnBu``  : Yellow -> Green -> Blue sequential"
    - ``PuBuGn``  : Purple -> Blue -> Green sequential"
    - ``BuGn``    : Blue -> Green sequential"
    - ``YlGn``    : Yellow -> Green sequential"

    Diverging
    ---------
    - ``PiYG``     : Pink -> Yellow -> Green"
    - ``PRGn``     : Purple -> Rose -> Green"
    - ``BrBG``     : Brown -> Beige -> Green"
    - ``PuOr``     : Orange -> White -> Purple"
    - ``RdGy``     : Red -> Grey -> Blue"
    - ``RdBu``     : Red -> White -> Blue"
    - ``RdYlBu``   : Red -> Yellow -> Blue"
    - ``RdYlGn``   : Red -> Yellow -> Green"
    - ``Spectral`` : Red -> Orange -> Blue -> Green"
    - ``coolwarm`` : Blue -> Grey -> Red"
    - ``bwr``      : Blue -> White -> Red"
    - ``seismic``  : Blue -> White -> Red"

    Qualitative
    -----------
    - ``Pastel1`` : Pastel qualitative palette"
    - ``Pastel2`` : Second pastel qualitative palette"
    - ``Paired``  : Paired qualitative palette"
    - ``Accent``  : Accent qualitative palette"
    - ``Dark2``   : Dark qualitative palette"
    - ``Set1``    : Distinct qualitative palette"
    - ``Set2``    : Second distinct qualitative palette"
    - ``Set3``    : Third distinct qualitative palette"
    - ``tab10``   : Tableau 10-color palette"
    - ``tab20``   : Tableau 20-color palette"
    - ``tab20b``  : Tableau 20-color bright palette"
    - ``tab20c``  : Tableau 20-color medium palette"

    Miscellaneous
    -------------
    - ``terrain``   : Terrain-like colormap"
    - ``ocean``     : Ocean-like colormap"
    - ``cubehelix`` : Cube helix colormap"
    - ``rainbow``   : Rainbow colormap"
    - ``twilight``  : Twilight colormap"

    Examples
    --------
        >>> from pyfmto.utilities.stroptions import Cmaps
        >>> print(Cmaps.viridis)
        viridis
        >>> plot_func_2d(func, lb, ub, dim, cmap=Cmaps.viridis)
    """

    viridis = "[Uniform] Purple -> Yellow (colorblind-friendly)"
    plasma = "[Uniform] Purple -> Pink -> Yellow"
    inferno = "[Uniform] Black -> Red -> Yellow"
    magma = "[Uniform] Black -> Purple -> White"
    cividis = "[Uniform] Colorblind-friendly version of viridis"

    Greys = "[Sequential] White to black grayscale"
    Purples = "[Sequential] White to purple sequential colormap"
    Blues = "[Sequential] White to blue sequential colormap"
    Greens = "[Sequential] White to green sequential colormap"
    Oranges = "[Sequential] White to orange sequential colormap"
    Reds = "[Sequential] White to red sequential colormap"
    YlOrBr = "[Sequential] Yellow -> Orange -> Brown sequential"
    YlOrRd = "[Sequential] Yellow -> Orange -> Red sequential"
    OrRd = "[Sequential] Orange -> Red sequential"
    PuRd = "[Sequential] Purple -> Red sequential"
    RdPu = "[Sequential] Red -> Purple sequential"
    BuPu = "[Sequential] Blue -> Purple sequential"
    GnBu = "[Sequential] Green -> Blue sequential"
    PuBu = "[Sequential] Purple -> Blue sequential"
    YlGnBu = "[Sequential] Yellow -> Green -> Blue sequential"
    PuBuGn = "[Sequential] Purple -> Blue -> Green sequential"
    BuGn = "[Sequential] Blue -> Green sequential"
    YlGn = "[Sequential] Yellow -> Green sequential"

    PiYG = "[Diverging] Pink -> Yellow -> Green"
    PRGn = "[Diverging] Purple -> Rose -> Green"
    BrBG = "[Diverging] Brown -> Beige -> Green"
    PuOr = "[Diverging] Orange -> White -> Purple"
    RdGy = "[Diverging] Red -> Grey -> Blue"
    RdBu = "[Diverging] Red -> White -> Blue"
    RdYlBu = "[Diverging] Red -> Yellow -> Blue"
    RdYlGn = "[Diverging] Red -> Yellow -> Green"
    Spectral = "[Diverging] Red -> Orange -> Blue -> Green"
    coolwarm = "[Diverging] Blue -> Grey -> Red"
    bwr = "[Diverging] Blue -> White -> Red"
    seismic = "[Diverging] Blue -> White -> Red"

    Pastel1 = "[Qualitative] Pastel qualitative palette"
    Pastel2 = "[Qualitative] Second pastel qualitative palette"
    Paired = "[Qualitative] Paired qualitative palette"
    Accent = "[Qualitative] Accent qualitative palette"
    Dark2 = "[Qualitative] Dark qualitative palette"
    Set1 = "[Qualitative] Distinct qualitative palette"
    Set2 = "[Qualitative] Second distinct qualitative palette"
    Set3 = "[Qualitative] Third distinct qualitative palette"
    tab10 = "[Qualitative] Tableau 10-color palette"
    tab20 = "[Qualitative] Tableau 20-color palette"
    tab20b = "[Qualitative] Tableau 20-color bright palette"
    tab20c = "[Qualitative] Tableau 20-color medium palette"

    terrain = "[Miscellaneous] Terrain-like colormap"
    ocean = "[Miscellaneous] Ocean-like colormap"
    cubehelix = "[Miscellaneous] Cube helix colormap"
    rainbow = "[Miscellaneous] Rainbow colormap"
    twilight = "[Miscellaneous] Twilight colormap"


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


class SeabornStyles(StrOptions):
    """
    Seaborn style options.

    This class provides a set of predefined style names with their
    descriptions for use in seaborn plots.

    Available styles
    -----
    - ``darkgrid`` : Dark background with grid lines (default seaborn style)
    - ``whitegrid`` : White background with grid lines
    - ``dark`` : Dark background with no grid lines
    - ``white`` : White background with no grid lines
    - ``ticks`` : White background with ticks on the axes

    Examples
    --------
        >>> from pyfmto.utilities.stroptions import SeabornStyles
        >>> print(SeabornStyles.whitegrid)
        whitegrid
        >>> sns.set_style(str(SeabornStyles.whitegrid))
    """

    darkgrid = "Dark background with grid lines (default seaborn style)"
    whitegrid = "White background with grid lines"
    dark = "Dark background with no grid lines"
    white = "White background with no grid lines"
    ticks = "White background with ticks on the axes"


class StrColors(StrOptions):
    black = 'black'
    gray = 'gray'
    red = 'red'
    green = 'green'
    yellow = 'yellow'
    blue = 'blue'
    magenta = 'magenta'
    cyan = 'cyan'
    white = 'white'
