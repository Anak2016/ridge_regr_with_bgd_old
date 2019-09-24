import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *
from models.utils import glorot, differentiate
from models.loss_func import mean_square_error


class MessagePassing():
    def __init__(self, data, loss_func=None, batch_size=None):
        self.register_params = {}
        self.loss_func = loss_func # loss_func
        self.bs = batch_size
        self.data = data

    def register(self, *args):
        '''register variable to be used in differentiate()'''
        for i,arg in enumerate(args):
            self.register_params[f'var_{i}'] = {'val':arg, 'diff_val': None} # is this the best datastrcuture to be used?
        # display2screen(self.register_params)

    def step(self, val, diff_val):
        return val - diff_val

    def backward_batch(self, loop_num):
        i = loop_num
        for k, v in self.register_params.items():
            if k.__name__.startswith('var_'):
                v['diff_val'] = differentiate(v['val'], self.loss_func)  # differentiate loss function wrt beta
            s = (i) * self.bs
            f = (i + 1) * self.bs if (i + 1) * self.bs < self.data.shape[0] + 1 else self.data.shape[0] + 1

            self.register_params['val'] = self.step(v['val'][s:f], v['diff_val'][s:f])  # update beta value

    def backward(self):
        '''
        does backpropagation
        :return: all data

        '''
        assert len(self.register_params) > 0, "no param has been registed "
        loop = 1 + self.data.shape[0]/ self.bs

        # TODO here>> how do i differentate value of loss_func per batch
        for i in range(0, loop ):
           self.backward_batch(i)

        return [v['val'] for k,v in self.register_params.items()]


class RidgeRegression(MessagePassing):
    ''' y = beta*X + intercept where b is intersect and A i slope of hyperplan'''
    def __init__(self, data=None, loss_func=None, batch_size=None, learning_rate = None):
        super(MessagePassing, self).__init__(loss_func, batch_size)

        self.beta = np.zeros((data.shape[0], data.shape[1]))
        self.intercept = np.zeros((data.shape[0], data.shape[1]))
        self.y = np.zeros((data.shape[0], data.shape[1]))
        self.x = data
        self.bs = batch_size
        self.lr = learning_rate

        #--------message_passing
        # self.message_passing = MessagePassing()
        self._initialize_parameters()
        # self.message_passing.register('a', self.beta) # register beta as a message
        self.register(self.beta) # register beta as a message

    def _initialize_parameters(self):
        self.beta = glorot(self.beta)


    def run(self):
        self.y = self.forward()
        self.beta = self.backward()

    def forward(self):
        return self.beta.dot(self.x) + self.intercept  # result



if __name__ == '__main__':
    from preprocessing import *

    WORKING_DIR = r"C:\Users\awannaphasch2016\PycharmProjects\ridge_regr_with_bgd"
    tmp = f'{WORKING_DIR}/datasets/Credit_N400_p9.csv'
    print(f"reading data from {tmp}...")
    data = pd.read_csv(tmp, sep=',').to_numpy().astype(object)
    data = preprocess_data(data)
    # display2screen(data)
    # check that data is standalized and centered
    mask = [True if not isinstance(i, str) else False for i in data[0, :]]  # mask for col index that is not categorized
    assert data[:, mask].mean(axis=0).sum() != 0, 'data is not centered '
    assert data[:, mask].astype(float).std(axis=0).sum() != 0, 'data is not stadalized'
    print("preprocessed data is completed !!")
    ridg_regr = RidgeRegression(data, mean_square_error, 32, 0.1)
    ridg_regr.run()