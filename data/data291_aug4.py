import os
from data import srdata

class DATA291_aug4(srdata.SRData):
    def __init__(self, args, name='DATA291_aug4', train=True, benchmark=False):
        super(DATA291_aug4, self).__init__(#calling thr parent class. init
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DATA291_aug4, self)._scan()#calling thr parent class. get file name
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DATA291_aug4, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, '291_HR')
        self.dir_lr = os.path.join(self.apath, '291_train_LR_bicubic')

