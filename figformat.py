import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
import numpy as np
import os
# FIGURE INITIAL SETTING 
style.use('default')
plt.rcParams['font.family'] ='arial' # Font
plt.rcParams["mathtext.fontset"] = "dejavuserif" # mathfont
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['legend.title_fontsize'] = 8
fontP = FontProperties()
fontP.set_size('x-small')


def get_A4size(margin_ratio=0.8, FSIZE = 7, msize=10,linewidth=0.5):
    centim = 1/2.54
    key = {'width':21.0 * centim * margin_ratio,
           'half':10.5 * centim * margin_ratio,
           'twothird':15.7 * centim * margin_ratio,
           'height':29.7 * centim * margin_ratio,
           'FSIZE': FSIZE,
           'msize': msize,
           'linewidth':linewidth
          }
    return key

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def subplot2gridlist(row,col,figsize):
    fig = plt.figure(figsize=figsize)
    axes = []
    for i in range(col):
        col_list = []
        for j in range(row):
            ax = plt.subplot2grid( (row,col), (j,i), rowspan=1,colspan=1)
            col_list.append(ax)
        axes.append(col_list)
    return fig, axes

def addlabels2axes(axes,position=(-0.1, 1.05),order='C',lowercase=True,fontsize=8):
    import string
    for i,ax in enumerate(np.array(axes).flatten(order=order)):
        if lowercase:
            ax.text(*position, string.ascii_lowercase[i], transform=ax.transAxes, size=fontsize, weight='bold')
        else:
            ax.text(*position, string.ascii_uppercase[i], transform=ax.transAxes, size=fontsize, weight='bold')

def adjustTicks(axes,xlim,ylim,xticks=[],xlabels=[],yticks=[],ylabels=[],FSIZE=8):
    count = 0
    num_rows = len(axes[0])
    num_cols = len(axes)
    for i in range(num_rows): 
        row = i
        for j in range(num_cols):
            col = j
            ax = axes[j][i]
            ax.tick_params(axis='x',labelsize=FSIZE); ax.tick_params(axis='y',labelsize=FSIZE)
            ax.set_xticks(ticks=xticks,labels=xlabels,size=FSIZE); ax.set_yticks(ticks=yticks,labels=ylabels,size=FSIZE)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            if col != 0 and row+1 != num_rows:
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(ax.get_xticklabels(), visible=False)
            elif col == 0 and row+1 != num_rows:
                plt.setp(ax.get_xticklabels(), visible=False)
            elif col != 0 and row+1 == num_rows:
                plt.setp(ax.get_yticklabels(), visible=False)
            elif col == 0 and row+1 == num_rows:
                pass

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def legends(names,flip=False):
    figure_format_df = pd.DataFrame()
    figure_format_df['exp'] = names 
    # colors = colorlist(len(names),cmap=cmap)
    # markers = ['^','s','P','*','2','8','X','$\heartsuit$','o','<','h','>','x','$\clubsuit$','D','d',"v",'p','H','$\spadesuit$']
    markers = ['^','s','P','*','2','8','X','o','<','h','>','x','$\clubsuit$','D','d',"v",'p','H','$\spadesuit$']
    colors = sns.color_palette('colorblind',len(names)+1)
    # colors = ['#F72585','#B5179E','#F72585','#560BAD','#480CA8','#3A0CA3','#3F37C9','#F1C453','#4361EE','#4895EF','#4CC9F0','#2C699A','#048BA8','#560BAD','#83E377','#EFEA5A','#F72585','#EFEA5A',"#F1C453","#F29E4C",'#7209B7','#B5179E','#0DB39E']
    mlist_temp = []
    clist_temp = []
    if len(names) > len(markers):
        floornum = len(names)//len(markers)
        modnum = len(names)%len(markers)
        for i in range(floornum):
            mlist_temp += markers
        if modnum != 0:
            mlist_temp += markers[0:modnum]
    else:
        mlist_temp = markers[0:len(names)]
    if len(names) > len(colors):
        floornum = len(names)//len(colors)
        modnum = len(names)%len(colors)
        for i in range(floornum):
            clist_temp += colors
        if modnum != 0:
            clist_temp += colors[0:modnum]
    else:
        clist_temp = colors[0:len(names)]
    if flip:
        clist_temp.reverse()
    figure_format_df['marker'] = mlist_temp
    figure_format_df['color'] = clist_temp
    marker_onlyline = [",","1","2","3","4","+","x",'$\heartsuit$','$\clubsuit$']
    return figure_format_df, marker_onlyline

def colorlist(num,cmap='hls'):
#     c_array =['#F72585','#B5179E','#7209B7','#560BAD','#480CA8','#3A0CA3','#3F37C9','#4361EE','#4895EF','#4CC9F0','#2C699A','#048BA8','#0DB39E','#83E377','#EFEA5A',"#F1C453","#F29E4C"]
    c_array = sns.color_palette(cmap)
    cnum = len(c_array)
    clist = []
    for i in range(num):
        ind = i%cnum
        clist.append(c_array[ind])
    from colormap import rgb2hex
    for i,cname in enumerate(clist):
        color = np.array(cname) * 255
        color = color.astype(int).tolist()
        color = rgb2hex(*color)
        clist[i] = color
    return clist

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# source: https://github.com/kuk/log-progress
def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

