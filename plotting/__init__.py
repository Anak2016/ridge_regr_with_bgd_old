import sys,os
USER = os.environ['USERPROFILE']
#sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.my_utility import plot_figures
from utility_code.python_lib_essential import *

from sklearn.decomposition import PCA
def plot_pca(x):
    pca = PCA(n_components=2)
    output = pca.fit_transform(x)
    '''
            config = {
            "loss history": {
                'x_label': 'False Positive Rate',
                'y_label': 'loss values',
                'legend': [{"kwargs": {"loc": "lower right"}}],
                'plot': [{"args": [range(len(loss_hist)), loss_hist]}]
            },
            "accuracy history": {
                'x_label': 'False Positive Rate',
                'y_label': 'accuracy values',
                'legend': [{"kwargs": {"loc": "lower right"}}],
                'plot': [{"args": [range(len(train_acc_hist)), train_acc_hist]},
                         {"args": [range(len(test_acc_hist)), test_acc_hist]},
                         ]
            },
        }
    '''
    config = {
         'pca' : {
             'x_label': 'pc1',
             'y_label': "pc2",
             'legend': [{"kwargs": {"loc": "lower right"}}],
             'plot': [{'args': [output[:,0], output[:,1]]}],
             "plot_style": "scatter"
         }
    }

    plot_figures(config)


