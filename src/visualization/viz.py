#########imports#####

%pylab inline
import matplotlib.pyplot as plt


import os
from scipy.stats import ttest_rel
import copy
import pickle
import pandas as pd
import copy

import numpy as np
from numpy import linalg as LA
import scipy
from sklearn import preprocessing

from gensim.models import Word2Vec
from gensim import utils
from gensim.corpora import Dictionary
from gensim.models.lsimodel import LsiModel

from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr


#########PLOT########

fsize=22
fsize2=18

#colors=['#74c476','#31a354','#005a32','black']
markers=['o','^','v','D']
lss=['-','-.','--','-','-','-','-','-']

_color_unbiasedarea='#bdbdbd'
_color_unbiasedarea2='red'
_color_unbiasedborder='black'#'#737373'
_color_unbiasedborder2='red'
_color_unbiasedpnt='#636363'
_color_SG_male='red'
_color_eSG_male='black'
_color_SG_female='#6baed6'#3182bd
_color_eSG_female='#08519c'


size_cof = 1
min_x = -0.3
max_x = 0.4
max_y=11.3#*size_cof


_bias_expectation_fem = eSG_bias_expectation_fem_scaled['orig']
_bias_expectation_mas = eSG_bias_expectation_mas_scaled['orig']
_bias_expectation_fem2 = SG_cosine_bias_expectation_fem['orig']
_bias_expectation_mas2 = SG_cosine_bias_expectation_mas['orig']


fig, ax= plt.subplots(nrows=1, ncols=1 , figsize=(8, 6))


ax.axvline(x=-_bias_expectation_mas, c=_color_unbiasedborder, ls='-.', lw=3)
ax.axvline(x=+_bias_expectation_fem, c=_color_unbiasedborder, ls='-.', lw=3)
ax.fill_between([-_bias_expectation_mas,_bias_expectation_fem], 
                [0, 0],
                [max_y, max_y],
                facecolor=_color_unbiasedarea, alpha=0.7, interpolate=True)
ax.axvline(x=-_bias_expectation_mas2, c=_color_unbiasedborder2, ls='-.', lw=3)
ax.axvline(x=+_bias_expectation_fem2, c=_color_unbiasedborder2, ls='-.', lw=3)
#ax.fill_between([-_bias_expectation_mas2,_bias_expectation_fem2], 
#                [0, 0],
#                [max_y, max_y],
#                facecolor=_color_unbiasedarea, alpha=0.1, interpolate=True)

for _manip_i, _manip in enumerate(manipluated_list[1:]):

    _bias_changes = bias_changes_scaled[_manip]

    for i in range(0,len(_bias_changes)):
        x=_bias_changes[i][1]
        y=[(10-i)*size_cof, (10-i)*size_cof]
        ax.scatter(x, y, c=_color_eSG_male, s=90, marker='o')
        ax.annotate("", xy=(x[0], y[0]), xytext=(x[1], y[1]),
                    arrowprops={'arrowstyle': '<-, head_width=0.5', 'ls':'-', 'lw':3, 
                                'color': _color_eSG_male, 'shrinkB':10, 'shrinkA':5}
                   )
        _text=_bias_changes[i][0]
        #_text=_text[0].upper()+_text[1:]
        if i==0:
            ax.annotate(_text, xy=(x[0], y[0]), xytext=(x[0]-0.01, y[0]+0.3), fontsize=fsize2)
        elif i==1:
            ax.annotate(_text, xy=(x[0], y[0]), xytext=(x[0]-0.02, y[0]+0.3), fontsize=fsize2)
        elif i==2:
            ax.annotate(_text, xy=(x[0], y[0]), xytext=(x[0]-0.01, y[0]+0.3), fontsize=fsize2)
        elif i==3:
            ax.annotate(_text, xy=(x[0], y[0]), xytext=(x[0]-0.01, y[0]+0.3), fontsize=fsize2)
        elif i==4:
            ax.annotate(_text, xy=(x[0], y[0]), xytext=(x[0]-0.04, y[0]+0.3), fontsize=fsize2)
        elif i==5:
            ax.annotate(_text, xy=(x[0], y[0]), xytext=(x[0]-0.17, y[0]+0.3), fontsize=fsize2)
        elif i==6:
            ax.annotate(_text, xy=(x[0], y[0]), xytext=(x[0]-0.115, y[0]+0.3), fontsize=fsize2)
        elif i==7:
            ax.annotate(_text, xy=(x[0], y[0]), xytext=(x[0]-0.09, y[0]+0.3), fontsize=fsize2)
        elif i==8:
            ax.annotate(_text, xy=(x[0], y[0]), xytext=(x[0]-0.14, y[0]+0.3), fontsize=fsize2)
        elif i==9:
            ax.annotate(_text, xy=(x[0], y[0]), xytext=(x[0]-0.09, y[0]+0.3), fontsize=fsize2)
        else:
            ax.annotate(_text, xy=(x[0], y[0]), xytext=(x[0]-0.09, y[0]+0.3), fontsize=fsize2)
        
        x=_bias_changes[i][2]
        y=[(10-i-0.35)*size_cof, (10-i-0.35)*size_cof]
        ax.scatter(x, y, c=_color_SG_male, s=90, marker='o')
        ax.annotate("", xy=(x[0], y[0]), xytext=(x[1], y[1]),
                    arrowprops={'arrowstyle': '<-, head_width=0.5', 'ls':'-', 'lw':3, 
                                'color': _color_SG_male, 'shrinkB':10, 'shrinkA':5}
                   )


ax.tick_params(which='major', labelsize=fsize)
ax.set_xlabel('Male                 Gender Bias                Female', fontsize=fsize)
ax.get_yaxis().set_visible(False)
ax.set_xlim(min_x, max_x)
ax.set_ylim(0, max_y)
plt.gca().xaxis.grid(True)

fig.tight_layout()
plt.show()
fig.set_size_inches(8, 6)
fig.savefig('plots/job_recSG_change_swapped.pdf', dpi=100, transparent=True, bbox_inches='tight', pad_inches=0)


