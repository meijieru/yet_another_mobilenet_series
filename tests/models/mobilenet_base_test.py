import unittest
import torch

import models.mobilenet_base as mb


def random_running_stat(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.running_mean.data.copy_(torch.rand_like(m.weight))
        m.running_var.data.copy_(torch.rand_like(m.weight))


class InvertedResidualChannelsTest(unittest.TestCase):

    def _run_get_depthwise_bn(self, expand):
        inp, oup = 6, 5
        m = mb.InvertedResidualChannels(inp, oup, 1, [2, 4], [3, 5], expand,
                                        torch.nn.ReLU)
        bn_list = m.get_depthwise_bn()
        for bn in bn_list:
            assert isinstance(bn, torch.nn.BatchNorm2d)

    def testGetDepthwiseBnExpand(self):
        self._run_get_depthwise_bn(True)

    def testGetNamedDepthwiseBnExpand(self):
        inp, oup = 6, 5
        m = mb.InvertedResidualChannels(inp, oup, 1, [2, 4], [3, 5], True,
                                        torch.nn.ReLU)
        self.assertListEqual(list(m.get_named_depthwise_bn()),
                             ['ops.{}.1.1'.format(i) for i in range(2)])
        prefix = 'prefix'
        self.assertListEqual(
            list(m.get_named_depthwise_bn(prefix=prefix)),
            ['{}.ops.{}.1.1'.format(prefix, i) for i in range(2)])

    @unittest.skip('Do not search when not expand')
    def testGetDepthwiseBnNonExpand(self):
        self._run_get_depthwise_bn(False)


class InvertedResidualChannelsFusedTest(unittest.TestCase):

    def _run_get_depthwise_bn(self, expand):
        inp, oup = 6, 5
        m = mb.InvertedResidualChannelsFused(inp, oup, 1, [2, 4], [3, 5],
                                             expand, torch.nn.ReLU)
        bn_list = m.get_depthwise_bn()
        for bn in bn_list:
            assert isinstance(bn, torch.nn.BatchNorm2d)

    def testGetDepthwiseBnExpand(self):
        self._run_get_depthwise_bn(True)

    def testGetNamedDepthwiseBnExpand(self):
        inp, oup = 6, 5
        m = mb.InvertedResidualChannelsFused(inp, oup, 1, [2, 4], [3, 5], True,
                                             torch.nn.ReLU)
        self.assertListEqual(list(m.get_named_depthwise_bn()),
                             ['depth_ops.{}.1.1'.format(i) for i in range(2)])
        prefix = 'prefix'
        self.assertListEqual(
            list(m.get_named_depthwise_bn(prefix=prefix)),
            ['{}.depth_ops.{}.1.1'.format(prefix, i) for i in range(2)])


if __name__ == "__main__":
    unittest.main()
