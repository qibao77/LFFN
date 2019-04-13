import os
import glob
import random
from data import common
import pickle
import numpy as np
import imageio
from helper import utilty as util
import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, flags, name='', train=True, benchmark=False):
        self.flags = flags
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = flags.scale
        self.idx_scale = 0

        data_range = [r.split('-') for r in flags.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if flags.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))

        self._set_filesystem(flags.train_dir)  # get dataset file path (override in son class)

        if flags.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, self.flags.scale_bin)
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()      # get all images path names (override in son class)

        if flags.ext.find(self.flags.scale_bin) >= 0:
            # Binary files are stored in 'bin' folder
            # If the binary file exists, load it. If not, make it.
            list_hr, list_lr = self._scan()

            self.images_hr = self._check_and_load(
                flags.ext, list_hr, self._name_hrbin()
            )

            self.images_lr = [
                self._check_and_load(flags.ext, l, self._name_lrbin(s)) \
                for s, l in zip(self.scale, list_lr)
            ]
        else:
            if flags.ext.find('img') >= 0 or benchmark:
                self.images_hr, self.images_lr = list_hr, list_lr
            elif flags.ext.find('sep') >= 0:
                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )
                for s in self.scale:
                    os.makedirs(
                        os.path.join(
                            self.dir_lr.replace(self.apath, path_bin),
                            'X{}'.format(s)
                        ),
                        # self.dir_lr.replace(self.apath, path_bin),
                        exist_ok=True
                    )
                # get all .pt images path names
                # self.images_hr, self.images_lr = [], [[] for _ in self.scale]
                self.images_hr, self.images_lr = [], []
                for h in list_hr:
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    self.images_hr.append(b)
                    self._check_and_load(
                        flags.ext, [h], b, verbose=True, load=False
                    )

                for l in list_lr:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr.append(b)
                    self._check_and_load(
                        flags.ext, [l], b, verbose=True, load=False
                    )
                    # for l in ll:
                    #     b = l.replace(self.apath, path_bin)
                    #     b = b.replace(self.ext[1], '.pt')
                    #     self.images_lr[i].append(b)
                    #     self._check_and_load(
                    #         flags.ext, [l], b,  verbose=True, load=False
                    #     )

        if train:
            self.repeat \
                = flags.test_every // (len(self.images_hr) // flags.batch_num)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, 'X{}'.format(self.scale), '*' + self.ext[0]))
        )
        # names_lr = [[] for _ in self.scale]
        # for f in names_hr:
        #     filename, _ = os.path.splitext(os.path.basename(f))
        #     for si, s in enumerate(self.scale):
        #         names_lr[si].append(os.path.join(
        #             self.dir_lr, 'X{}/{}x{}{}'.format(
        #                 s, filename, s, self.ext[1]
        #             )
        #         ))

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.pt'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.pt'.format(self.split, scale)
        )

    def _check_and_load(self, ext, l, f, verbose=True, load=True):
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f: ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))
            b = [{
                'name': os.path.splitext(os.path.basename(_l))[0],
                'image': imageio.imread(_l)
            } for _l in l]
            with open(f, 'wb') as _f: pickle.dump(b, _f)
            return b

    def __getitem__(self, idx):
        scale = self.scale[self.idx_scale]
        if self.train:
            new_idx = random.randint(0, self.end)
            lr, hr, filename = self._load_file(new_idx)
            while lr.shape[0] < self.flags.batch_image_size or lr.shape[1] < self.flags.batch_image_size:
                new_idx = random.randint(0, self.end)
                lr, hr, filename = self._load_file(new_idx)
        else:
            lr, hr, filename = self._load_file(idx)

        if self.train:
            hr, lr = common.random_crop(hr, lr, self.flags.batch_image_size, scale)  # LR patch size
            lr, hr = common.random_flip_and_rotate(im1=lr, im2=hr)
        lr = util.resize_image_by_pil(hr, 1 / scale)
        # lr, hr = self.get_patch(lr, hr)

        lr, hr = common.set_channel(lr, hr, n_channels=self.flags.channels)
        lr_tensor, hr_tensor = torch.from_numpy(lr/255.-0.5).float(), torch.from_numpy(hr/255.-0.5).float()
        # lr_tensor, hr_tensor = common.np2Tensor(
        #     lr, hr, rgb_range=self.flags.rgb_range
        # )

        lr_tensor, hr_tensor = lr_tensor.view(1, self.flags.batch_image_size, self.flags.batch_image_size, self.flags.channels).squeeze(),\
                               hr_tensor.view(1, self.flags.batch_image_size*scale, self.flags.batch_image_size*scale, self.flags.channels).squeeze()

        return lr_tensor, hr_tensor, filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]

        if self.flags.ext.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.flags.ext == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
                lr = imageio.imread(f_lr)
            elif self.flags.ext.find('sep') >= 0:
                with open(f_hr, 'rb') as _f: hr = np.load(_f)[0]['image']
                with open(f_lr, 'rb') as _f: lr = np.load(_f)[0]['image']

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.flags.batch_image_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.flags.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

