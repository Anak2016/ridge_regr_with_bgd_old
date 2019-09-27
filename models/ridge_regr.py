import sys,os
USER = os.environ['USERPROFILE']
#sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *
from models.utils import glorot, differentiate, cross_validation
# from models.loss_func import mean_square_error
from models.loss_func import MSE
from arg_parser import *
from data import CreditData

class MessagePassing():
    def __init__(self, x, y, loss_func=None, batch_size=None, epochs=None, *args, **kwargs):
        self.register_params = {}
        self.y = y
        self.loss_func = loss_func # loss_func
        self.bs = batch_size
        self.data = x
        self.epochs = epochs
        self.loss_val_hist = []
        self.__args__ = args
        self.__kwargs__ = kwargs
        # display2screen(len(self.__args__), self.__kwargs__)

    def register(self, **kwargs):
        '''register variable to be used in differentiate()'''
        for i,(k,v) in enumerate(kwargs.items()):
            self.register_params[f'var_{i}'] = {'name': k,'val':v, 'diff_val': None} # is this the best datastrcuture to be used?
        # display2screen(self.register_params)

    # def get_loss_val(self, loop_num, all_params=None):
    #     '''get rid of all_params'''
    #     all_params = {name: val for name, val in all_params.items()}
    #     loss_val = self.loss_func.run(loop_num).subs([(name, val) for name, val in all_params.items()])
    #     return loss_val

    def get_loss_val(self, loop_num):
        '''refactor code get_loss_val and differentiate FOR NOW THIS IS FOR READBILIT AND DEBUGGIN'''
        all_params = {f"{v['name']}{i}": v['val'][i] for k, v in self.register_params.items() for i in
                                  range(0, v['val'].shape[0])}
        # all_params = {name: val for name, val in all_params.items()}
        loss_val = self.loss_func.run_batch(loop_num).subs([(name, val) for name, val in all_params.items()])
        return loss_val

    def step(self, val, diff_val):
        return val - diff_val * args.lr
    def write2file(self,save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savetxt(save_path,np.array(self.loss_val_hist).squeeze())


    def backward_batch(self, loop_num, epoch_num, cv_num):
        i = loop_num
        # all_params = {f"{v['name']}{i}" : {"val":v['val'][i], "diff_val":None} for k,v in self.register_params.items() for i in range(0,v['val'].shape[0])}
        # all_params_no_diff_val = {f"{v['name']}{i}" : v['val'][i]for k,v in self.register_params.items() for i in range(0,v['val'].shape[0])}

        # print(f"epoch_num={epoch_num} loop={loop_num}; val= {self.register_params['var_0']['val']}")

        for k, v in self.register_params.items():
            if k.startswith('var_'):
                #TODO here>>  create derivative equation of diff_val
                #       >> why is it diverging??
                #       >> did i normalized this?
                # v['diff_val'] =   2 * (self.data * np.expand_dims(self.data.dot(v["val"]), axis=1) + 2 * args.lmda * v["val"].squeeze()).sum(axis =0)
                v['diff_val'] =  - (2 * (self.y - data.dot(v["val"].T)).dot(data) + 2 * args.lmda * v['val']) # this is wrong
                # v['diff_val'] = np.empty_like(v['diff_val']).astype(np.float64)
                #                 # print(v['diff_val'])
                self.register_params[k]['val'] = self.step(v['val'], v['diff_val'])  # update beta value

        loss_val = np.array(self.get_loss_val(loop_num)).squeeze()
        self.loss_val_hist.append(loss_val)
        self.loss_val = loss_val


        if args.verbose:
            print(f'cv={cv_num}, epoch={epoch_num}, batch={self.bs}'
                  f'    ==> {i * self.bs}: loss_val={loss_val}')

    # def backward_batch(self, loop_num, epoch_num, cv_num):
    #     i = loop_num
    #     all_params = {f"{v['name']}{i}" : {"val":v['val'][i], "diff_val":None} for k,v in self.register_params.items() for i in range(0,v['val'].shape[0])}
    #     # print(f"epoch_num={epoch_num} loop={loop_num}; val= {self.register_params['var_0']['val']}")
    #     all_params_no_diff_val = {f"{v['name']}{i}" : v['val'][i]for k,v in self.register_params.items() for i in range(0,v['val'].shape[0])}
    #
    #     for k, v in self.register_params.items():
    #         if k.startswith('var_'):
    #             reconstruct_diff_vals = []
    #             for param,value in all_params_no_diff_val.items():
    #                 all_params[param]['diff_val'] = differentiate(self.loss_func, loop_num, learned_param_name = param, all_params = all_params_no_diff_val)
    #                 reconstruct_diff_vals.append(all_params[param]['diff_val'])
    #
    #             array_diff_val = np.array(reconstruct_diff_vals)
    #             # print(array_diff_vals)
    #             self.register_params[k]['diff_val'] =  array_diff_val.squeeze()
    #
    #             # TODO here>> construct diff_val from beta0 to beta10 as array
    #             #  > look at plot of loss_val vs different lambda (penality coefficient)
    #             #  > look at plot of loss_val vs leraning rate
    #             #  > try to speed up the derivative  process by skipping derivative step
    #             #  > implement cross validation
    #             #  > plot loss val vs epoch
    #             #  > plot beta vs lamda (penality coefficient)
    #             #  > why loss_val increse each run?
    #             #        >> why is there always a jump of loss_val between epochs
    #             #        >> why is self.register_params[k]['val'] repeat same score sequence from batch after certain epoch??
    #             #        >> diff_val is alot higher than val after derivation
    #             #  > check why loss_val is not steadily decreasing; do i step in the right direction??
    #             #  > run backward_batch for many
    #             #  > cross validation
    #             #  > plot loss_val over time
    #             self.register_params[k]['val'] = self.step(v['val'], v['diff_val'])  # update beta value
    #     if epoch_num == 1:
    #         pass
    #     loss_val = np.array(self.get_loss_val(loop_num, all_params = all_params_no_diff_val)).squeeze()
    #     # print(f"labels = {self.y[:10]} prediction = {(self.data.dot(self.register_params[k]['val']))[:10]}")
    #     print(f"distance = {(self.y - self.data.dot(self.register_params[k]['val'])).sum()}")
    #     self.loss_val_hist.append(loss_val)
    #     if args.verbose:
    #         print(f'cv={cv_num}, epoch={epoch_num}, batch={self.bs} index '
    #               f'    ==> {i * self.bs}: loss_val={loss_val}')



    def backward(self, epoch, cv_num, beta):
        '''
        does backpropagation
        :return: all data

        '''
        print('doing backward..')
        assert len(self.register_params) > 0, "no param has been registed "
        batch_loop = 1 + int(self.data.shape[0]/ self.bs) if self.data.shape[0]/ self.bs   != int(self.data.shape[0]/ self.bs) + 1 else int(self.data.shape[0])

        for i in range(0, batch_loop ):
            self.backward_batch(i, epoch, cv_num)

        # --------write2file
        save_path = f'/log/loss_val/bs={args.bs}_epochs={args.epochs}_cv={args.cv}/loss_val_lr={args.lr}_lmda={args.lmda}.txt'
        self.write2file(save_path)

        # self.loss_val
        self.output = {v['name']: v['val'] for k,v in self.register_params.items()}
        self.output.update({'loss_val':self.loss_val})


        return self.output
        # return {'loss_val': self.loss_func}.items()


class RidgeRegression(MessagePassing):
    ''' y = beta*X + intercept where b is intersect and A i slope of hyperplan'''
    def __init__(self, x=None,labels=None, loss_func=None, batch_size=None, learning_rate = None, epochs=None):
        '''

        :param data:
        :param loss_func: a class of loss function (not function or method)
        :param batch_size:
        :param learning_rate:
        '''
        # self.y = np.zeros((data.shape[0], 1)) # prediction
        self.y = labels
        self.x = x
        self.bs = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_val = None
        # --------message_passing
        # self.message_passing = MessagePassing()
        self._initialize_parameters()
        self.loss_func = loss_func( self.y,self.x,self.beta, lmda=args.lmda, l1=True, batch_size=self.bs)
        super(RidgeRegression, self).__init__(self.x, self.y, self.loss_func, batch_size, epochs)

        # self.message_passing.register('a', self.beta) # register beta as a message
        self.register(beta= self.beta.squeeze()) # register beta as a message
        # self.register(intercept=self.intercept)

    def _initialize_parameters(self):
        # self.beta = np.random.uniform((data.shape[1],))
        # self.beta = np.zeros((self.x.shape[0],self.x.shape[1],))
        self.beta = np.zeros((1,self.x.shape[1]+1))
        self.beta = glorot(self.beta)
        self.x = np.hstack((self.x,np.ones((self.x.shape[0],1))))
        # self.intercept = np.zeros((self.x.shape[0]))
        # self.intercept = glorot(self.beta)

    def train(self, x ,y, cv_num):
        self.x, self.y = x, y
        loss_epoch = []
        for epoch_num in range(0,self.epochs):
            ridg_regr.run_batch(epoch_num, cv_num)
            if args.report_performance:
                loss_epoch.append(self.loss_val)
                print(f'loss_val = {self.loss_val}')
                # report_performances()
        if args.report_performance:
            print(f'loss_val = {sum(loss_epoch)/len(loss_epoch)}')

    def pred(self, x, y):
        '''
        predict performance from test set
        :return:
        '''
        self.x, self.y = x, y
        self.output = self.apply_formular(x)
        self.pred_result = self.loss_func.run()
        if args.verbose:
            print(f'output = {self.output}')
        #--------compute accuracy
        if args.report_performance:
            print(f'loss_val = {self.loss_val}')

    def apply_formular(self,x):
        return self.x.dot(self.beta.T)

    def run_batch(self, epoch_num, cv_num):
        # self.forward()
        #TODO here>> check that beta is updated correctly after backward is completely.
        self.result = self.backward(epoch_num, cv_num, self.beta)
        self.update(self.result) # update value of beta

    def update(self, params_dict=None):
        for i,j in params_dict.items():
            if i == 'beta':
                self.beta = j
            if i == 'loss_val':
                self.loss_val = j
        # self.beta = [j for i, j in params_dict.items() if i == 'beta'][0]  # get beta value

    # def forward(self):
    #     print('doing forward..')
    #     #TODO here>> update beta in RidgeRegression
    #     # > directly compare prediction after 1 epoch. ( accuracy)
    #     self.y = self.x.dot(self.beta.T)
    #     # return self.x.dot(self.beta.T) + self.intercept  # result #



if __name__ == '__main__':
    from preprocessing import *

    WORKING_DIR = r"C:\Users\awannaphasch2016\PycharmProjects\ridge_regr_with_bgd"
    tmp = f'{USER}/PycharmProjects/ridge_regr_with_bgd/datasets/Credit_N400_p9.csv'
    print(f"reading data from {tmp}...")
    data = pd.read_csv(tmp, sep=',').to_numpy().astype(object)[:, 1:]
    credit_data = CreditData(data)
    credit_data.preprocess_data()
    x = credit_data.x
    labels= credit_data.y
    # display2screen(data)
    # check that data is standalized and centered
    # mask = [True if not isinstance(i, str) else False for i in x[0, :]]  # mask for col index that is not categorized
    mask = credit_data.mask
    assert x[:, mask].mean(axis=0).sum() != 0, 'data is not centered '
    assert x[:, mask].astype(float).std(axis=0).sum() != 0, 'data is not stadalized'
    print("preprocessed data is completed !!")

    if args.plot_pca:
        from plotting import *
        plot_pca(x)

    # def mean_square_error(y, x, beta, lmda=0.1, l1=True):
    # ridg_regr = RidgeRegression(data, MSE, 32, 0.1, epochs=args.epoch)
    # def __init__(self, data=None, loss_func=None, batch_size=None, learning_rate = None):

    ridg_regr = RidgeRegression(x=x,
                                labels=labels,
                                loss_func=MSE,
                                batch_size=args.bs,
                                learning_rate=args.lr,
                                epochs=args.epochs)

    cross_validation(x, labels, args.cv, ridg_regr)
    # epoch should be done here.
    # ridg_regr.run()
    #--------cross validation
    # cv = args.cv
    # test_size = x.size[0]/cv
    # tmp = [True for i in range(0,cv)]
    # s = 0
    # f = test_size
    # for i in range(0,cv):
    #     s = s + i * test_size
    #     f = s + test_size if s + test_size > x.size[0] + 1 else x.size[0] + 1
    #     test_mask = [True if (i < f and i >=f) else False for i in range(0,x.size[0]) ]
    #     train_mask = [not i for i in test_mask]
    #     x_train, y_train, x_test, y_test = x[train_mask], labels[train_mask], x[test_mask], labels[test_mask]
    #     #--------trian
    #     ridg_regr.train(x_train, labels.trian)
    #     #--------predict
    #     ridg_regr.predict(x_test, labels.test)
