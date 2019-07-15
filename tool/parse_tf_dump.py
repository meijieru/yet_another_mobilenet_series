import argparse
import numbers
import re
import yaml
import pickle
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn

import models.mobilenet_supernet as ms
import models.mobilenet_base as mb

SKIP_MODULES = tuple([nn.ReLU, nn.ReLU6, nn.AvgPool2d, nn.Dropout])
CONV_PREFIX = 'single-path/mnas_net_model'
BN_PREFIX = 'single-path'


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def convert_module(m, prefix_torch=None, prefix_tf=None, idx=None):

    def fix_name(prefix, names, split):
        if prefix is None:
            return names

        if isinstance(names, str):
            names = split.join([prefix, names])
        elif isinstance(names, list):
            names = [fix_name(prefix, name, split) for name in names]
        else:
            raise NotImplementedError()
        return names

    res = {}
    if isinstance(m, nn.Sequential):
        for i, sub_module in m.named_children():
            tmp = convert_module(sub_module, prefix_torch=str(i))
            res.update(tmp)
    elif isinstance(m, ms.InvertedResidual):
        assert isinstance(m.expand_ratio, numbers.Number)
        expand = m.expand_ratio != 1
        if not expand:
            idx_depth = 0
            idx_reduct = 1
            res[r'conv2d(_[0-9]+)?/kernel'] = 'ops.0.{}.weight'.format(
                idx_reduct)
        else:
            idx_depth = 1
            idx_reduct = 2
            res[r'conv2d(_[0-9]+)?/kernel'] = [
                'ops.0.0.0.weight', 'ops.0.{}.weight'.format(idx_reduct)
            ]
            res.update(
                convert_module(m.ops[0][0][1], idx=0, prefix_torch='ops.0.0.1'))
        depthwise_scope = r'depthwise_conv2d_masked(_[0-9]+)?/depthwise_kernel'
        res[depthwise_scope] = 'ops.0.{}.0.weight'.format(idx_depth)
        res.update(
            convert_module(m.ops[0][idx_depth][1],
                           idx=idx_depth,
                           prefix_torch='ops.0.{}.1'.format(idx_depth)))
        res.update(convert_module(m.pw_bn, idx=idx_reduct,
                                  prefix_torch='pw_bn'))
    elif isinstance(m, nn.Conv2d):
        res[r'conv2d(_[0-9]+)?/kernel'] = 'weight'
        if m.bias is not None:
            raise NotImplementedError()
    elif isinstance(m, nn.Linear):
        res[r'dense/kernel'] = 'weight'
        res[r'dense/bias'] = 'bias'
    elif isinstance(m, nn.BatchNorm2d):
        if idx == 0 or idx is None:
            extra = ''
        else:
            extra = '_{}'.format(idx)
        for name_tf, name_torch in [('moving_mean', 'running_mean'),
                                    ('moving_variance', 'running_var'),
                                    ('gamma', 'weight'), ('beta', 'bias')]:
            res[r'batch_normalization{}/{}'.format(extra, name_tf)] = name_torch
    elif isinstance(m, (nn.ReLU, nn.ReLU6)):
        pass
    else:
        raise NotImplementedError(m)
    res_final = {}
    for name_tf, name_torch in res.items():
        name_tf = fix_name(prefix_tf, name_tf, '/')
        name_torch = fix_name(prefix_torch, name_torch, '.')
        # print('Map {} -> {}'.format(name_tf, name_torch))
        res_final[name_tf] = name_torch
    return res_final


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', default='./exp/tf_checkpoint.pkl')
    parser.add_argument(
        '--model_config_path',
        default='./apps/mobilenet/models/mobilenet_v2_1.0_relu.yml')
    parser.add_argument('--dst_path', default='./exp/tf_checkpoint.pt')
    parser.add_argument('--prefix', default='module')
    parser.add_argument('--use_ema', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.src_path, 'rb') as f:
        dump_res = pickle.load(f)
    for key, var in sorted(dump_res.items()):
        print(key, var.shape, 0.5 * (var**2).sum())

    with open(args.model_config_path, 'r') as f:
        model_kwargs = yaml.safe_load(f)
        print(model_kwargs)

    # stem
    block_scope_remap = {
        'features.0': 'mnas_stem',
        'classifier.1': 'mnas_head',
        'features.18': 'mnas_head',
    }
    for i in range(1, 17 + 1):
        block_scope_remap['features.{}'.format(i)] = 'mnas_blocks_{}'.format(i -
                                                                             1)
    print(block_scope_remap)

    model = ms.mobilenet_v2(**model_kwargs)
    model.apply(mb.init_weights_mnas)

    tf_torch_name_remap = {}
    for name, m in model.features.named_children():
        if isinstance(m, SKIP_MODULES):
            continue
        block_name = 'features.{}'.format(name)
        tmp = convert_module(m,
                             prefix_torch=block_name,
                             prefix_tf=block_scope_remap[block_name])
        tf_torch_name_remap.update(tmp)
    for name, m in model.classifier.named_children():
        if isinstance(m, SKIP_MODULES):
            continue
        block_name = 'classifier.{}'.format(name)
        tmp = convert_module(m,
                             prefix_torch=block_name,
                             prefix_tf=block_scope_remap[block_name])
        tf_torch_name_remap.update(tmp)
    print('tf_torch_name_remap')
    pprint(tf_torch_name_remap)

    checkpoints = {}
    sorted_keys = sorted(dump_res.keys(), key=natural_keys)
    for name_tf, name_torch in tf_torch_name_remap.items():
        if args.use_ema:
            suffix = r'/ExponentialMovingAverage'
        else:
            suffix = r'$'
        pattern = re.compile(r'{}{}'.format(name_tf, suffix))
        match_keys = []
        for key in sorted_keys:
            match = pattern.search(key)
            if match:
                match_keys.append(key)

        if isinstance(name_torch, str):
            assert len(match_keys) == 1, '{} match: {}'.format(
                name_tf, match_keys)
            checkpoints[name_torch] = match_keys[0]
        elif isinstance(name_torch, list):
            assert len(name_torch) == len(match_keys)
            for nt, key_match in zip(name_torch, match_keys):
                checkpoints[nt] = key_match
        else:
            raise ValueError()
    print('checkpoints')
    pprint(checkpoints)

    torch_names = set(model.state_dict().keys())
    for key in checkpoints.keys():
        assert key in torch_names, key
        torch_names.discard(key)
    for name_remain in torch_names:
        assert 'num_batches_tracked' in name_remain

    state_dict = model.state_dict()
    TEST_KEY = 'features.16.ops.0.1.0.weight'
    print((state_dict[TEST_KEY]**2).sum())
    for name_torch, name_tf in checkpoints.items():
        var = state_dict[name_torch]
        var_tf = dump_res[name_tf]
        # print(name_tf, name_tf, var.size(), var_tf.shape)
        assert var.dim() == var_tf.ndim
        if var.dim() == 1:
            pass
        elif var.dim() == 2:  # linear weight
            var_tf = var_tf.T
        elif var.dim() == 4:  # conv weight
            kh, kw, c_in, c_out = var_tf.shape
            if c_out == 1 and var.size(1) == 1:  # depthwise conv
                var_tf = np.transpose(var_tf, (2, 3, 0, 1))
            else:
                var_tf = np.transpose(var_tf, (3, 2, 0, 1))
        else:
            raise ValueError()
        var.copy_(torch.from_numpy(var_tf))

    print(checkpoints[TEST_KEY])
    print((state_dict[TEST_KEY]**2).sum())
    if args.prefix:
        state_dict = {
            '{}.{}'.format(args.prefix, key): val
            for key, val in state_dict.items()
        }
    torch.save(state_dict, args.dst_path)
