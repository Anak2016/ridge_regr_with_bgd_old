import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *
from parameters import *

from preprocessing import *

def centering(x):
    '''

    :param x: numpy data ; dim = n*m where n = # instances and m = * features
    :return:
    '''
    return x - x.astype(float).mean(axis=0)
def standalize(x):
    '''

    :param x:  numpy data ; dim = n*m where n = # instances and m = * features
    :return:
    '''
    return x / x.astype(float).std(axis=0)

def preprocess_data(x):
    '''

    :param x:
    :return: return centering + standalize data
    '''
    mask = [True if not isinstance(i,str) else False for i in x[0,:]]# mask for col index that is not categorized
    x[:, mask] = centering(x[:, mask])
    x[:, mask] = standalize(x[:, mask])
    return x

if __name__ == '__main__':
    WORKING_DIR = r"C:\Users\awannaphasch2016\PycharmProjects\ridge_regr_with_bgd"
    tmp = f'{WORKING_DIR}/datasets/Credit_N400_p9.csv'
    print(f"reading data from {tmp}...")
    data = pd.read_csv(tmp, sep=',').to_numpy().astype(object)
    data = preprocess_data(data)
    # display2screen(data)
    #check that data is standalized and centered
    mask = [True if not isinstance(i, str) else False for i in data[0, :]]  # mask for col index that is not categorized
    assert data[:, mask].mean(axis=0).sum() != 0, 'data is not centered '
    assert data[:, mask].astype(float).std(axis=0).sum() != 0 , 'data is not stadalized'
    print("preprocessed data is completed !!")


