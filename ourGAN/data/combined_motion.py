from torch.utils.data import Dataset
import copy
from data.motion_dataset import MotionData
import os
import numpy as np
import torch
from data.bvh_parser import BVH_file
from data import get_test_set
import sys
sys.path.append("..")
from option_parser import get_std_bvh


class MixedData0(Dataset):
    """
    Mixed data for many skeletons but one topologies
    """
    def __init__(self, args, motions, skeleton_idx):
        super(MixedData0, self).__init__()
        self.motions = motions
        self.motions_reverse = torch.tensor(self.motions.numpy()[..., ::-1].copy())
        self.skeleton_idx = skeleton_idx
        self.length = motions.shape[0]
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.args.data_augment == 0 or torch.rand(1) < 0.5:
            return [self.motions[item], self.skeleton_idx[item]]
        else:
            return [self.motions_reverse[item], self.skeleton_idx[item]]


class MixedData(Dataset):
    """
    data_gruop_num * 2 * samples
    """
    def __init__(self, args, datasets_groups):
        # device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.final_data = []
        self.length = 0
        self.offsets = []
        # self.joint_topologies = []
        self.character_num = len(datasets_groups)
        seed = 19260817
        all_datas = []
        # for datasets in datasets_groups:
        for i, dataset in enumerate(datasets_groups):
            new_args = copy.copy(args)
            new_args.data_augment = 0
            new_args.dataset = dataset

            all_datas.append(MotionData(new_args))
            file = BVH_file(get_std_bvh(dataset=dataset))
            if i == 0:
                self.joint_topology = file.topology
            new_offset = file.get_normalize_offset()
            new_offset = torch.tensor(new_offset, dtype=torch.float)
            new_offset = new_offset.reshape((1,) + new_offset.shape)
            self.offsets.append(new_offset)
        self.offsets = torch.cat(self.offsets, dim=0)
        # self.offsets = self.offsets.to(device)

        pt = 0
        for datasets in all_datas:
            skeleton_idx = []
            # print(datasets.data.shape)
            skeleton_idx += [pt]*len(datasets)
            pt += 1
            # print(skeleton_idx)
            if self.length != 0 and self.length != len(skeleton_idx):
                self.length = min(self.length, len(skeleton_idx))
            else:
                self.length = len(skeleton_idx)
                # print("this len: ", self.length)
            self.final_data.append(MixedData0(args, datasets.data, skeleton_idx))


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        res = []
        for data in self.final_data:
            res.append(data[item])
        return res


class TestData(Dataset):
    def __init__(self, args, characters):
        self.characters = characters
        self.file_list = get_test_set()
        self.mean = []
        self.joint_topologies = []
        self.var = []
        self.offsets = []
        self.ee_ids = []
        self.args = args
        self.device = torch.device(args.cuda_device)

        for i, character_group in enumerate(characters):
            # print(i, character_group)
            mean_group = []
            var_group = []
            offsets_group = []
            for j, character in enumerate(character_group):
                # print(j, character)
                file = BVH_file(get_std_bvh(dataset = character))
                # print(file)
                if j == 0:
                    self.joint_topologies.append(file.topology)
                    self.ee_ids.append(file.get_ee_id())
                new_offset = file.offset
                new_offset = torch.tensor(new_offset, dtype=torch.float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)
                offsets_group.append(new_offset)
                mean = np.load('./datasets/Mixamo/mean_var/{}_mean.npy'.format(character))
                var = np.load('./datasets/Mixamo/mean_var/{}_var.npy'.format(character))
                mean = torch.tensor(mean)
                mean = mean.reshape((1, ) + mean.shape)
                var = torch.tensor(var)
                var = var.reshape((1, ) + var.shape)
                mean_group.append(mean)
                var_group.append(var)
                # print("new_offset: ", new_offset.shape, new_offset)

            mean_group = torch.cat(mean_group, dim=0).to(self.device)
            var_group = torch.cat(var_group, dim=0).to(self.device)
            offsets_group = torch.cat(offsets_group, dim=0).to(self.device)
            self.mean.append(mean_group)
            self.var.append(var_group)
            self.offsets.append(offsets_group)

    def __getitem__(self, item):
        res = []
        bad_flag = 0
        for i, character_group in enumerate(self.characters):
            res_group = []
            ref_shape = None
            for j in range(len(character_group)):
                new_motion = self.get_item(i, j, item)
                if new_motion is not None:
                    new_motion = new_motion.reshape((1, ) + new_motion.shape)
                    new_motion = (new_motion - self.mean[i][j]) / self.var[i][j]
                    ref_shape = new_motion
                res_group.append(new_motion)

            if ref_shape is None:
                print('Bad at {}'.format(item))
                return None
            for j in range(len(character_group)):
                if res_group[j] is None:
                    bad_flag = 1
                    res_group[j] = torch.zeros_like(ref_shape)
            if bad_flag:
                print('Bad at {}'.format(item))

            res_group = torch.cat(res_group, dim=0)
            res.append([res_group, list(range(len(character_group)))])
        return res

    def __len__(self):
        return len(self.file_list)

    def get_item(self, gid, pid, id):
        character = self.characters[gid][pid]
        path = './datasets/Mixamo/{}/'.format(character)
        if isinstance(id, int):
            file = path + self.file_list[id]
        elif isinstance(id, str):
            file = id
        else:
            raise Exception('Wrong input file type')
        if not os.path.exists(file):
            raise Exception('Cannot find file')
        print("combined_motion, file_path: ", file)
        file = BVH_file(file)
        motion = file.to_tensor(quater=self.args.rotation == 'quaternion')
        # print("combined motion, motion shape: {}, motion: {}" .format(motion.shape, motion))
        motion = motion[:, ::2]
        length = motion.shape[-1]
        length = length // 4 * 4
        return motion[..., :length].to(self.device)

    def denorm(self, gid, pid, data):
        means = self.mean[gid][pid, ...]
        var = self.var[gid][pid, ...]
        return data * var + means

    def normalize(self, gid, pid, data):
        means = self.mean[gid][pid, ...]
        var = self.var[gid][pid, ...]
        return (data - means) / var
