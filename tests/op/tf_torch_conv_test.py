import unittest
import numpy as np
import torch
import tensorflow as tf
import models.mobilenet_base as mb


def assertAllClose(a, b, rtol=1e-6, atol=1e-6, msg=None):
    msg = msg or ''
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def nhwc2nchw(x):
    return np.transpose(x, (0, 3, 1, 2))


class Conv2dTest(unittest.TestCase):

    def _test_valid(self, stride=1):
        y = np.random.rand(1, 100, 100, 1).astype(np.float32)
        filterx = np.random.rand(4, 5, 1, 2).astype(np.float32)
        a = tf.nn.conv2d(y, filterx, stride, 'VALID')
        with tf.Session() as sess:
            t = (sess.run(a))

        x = torch.nn.Conv2d(1, 2, (4, 5), stride=stride, bias=False)
        filter = np.transpose(filterx, (3, 2, 0, 1))
        x.weight.data.copy_(torch.nn.Parameter(torch.from_numpy(filter)))
        z = nhwc2nchw(y)
        l = x(torch.from_numpy(z)).detach().numpy()

        assertAllClose(nhwc2nchw(t), l)

    def _test_same(self, stride=1):
        var_tf = np.random.rand(3, 3, 3, 32).astype(np.float32)
        x_tf = np.ones((1, 224, 224, 3), dtype=np.float32)
        x = np.transpose(x_tf, (0, 3, 1, 2))

        res_tf = tf.nn.conv2d(x_tf, var_tf, stride, 'SAME')
        with tf.Session() as sess:
            res_tf = (sess.run(res_tf))

        m = mb.Conv2dSame(3, 32, 3, stride=stride, bias=False)
        m.weight.data.copy_(torch.from_numpy(np.transpose(var_tf,
                                                          (3, 2, 0, 1))))
        res = m(torch.from_numpy(x)).detach().numpy()

        assertAllClose(res, np.transpose(res_tf, (0, 3, 1, 2)))

    def test_valid_padding(self):
        self._test_valid(1)
        self._test_valid(2)

    def test_same_padding(self):
        self._test_same(1)
        self._test_same(2)


class BatchNorm2dTest(unittest.TestCase):

    def _test(self, x, batch_norm_momentum, batch_norm_epsilon, fused, mode):
        op_tf = tf.layers.BatchNormalization(axis=3,
                                             momentum=batch_norm_momentum,
                                             epsilon=batch_norm_epsilon,
                                             fused=fused)
        op_tf = op_tf(x, mode == 'train')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            rhs = sess.run(op_tf)

        device_torch = 'cuda'
        x_torch = nhwc2nchw(x)
        op_torch = torch.nn.BatchNorm2d(x_torch.shape[1],
                                        momentum=1.0 - batch_norm_momentum,
                                        eps=batch_norm_epsilon).to(device_torch)
        op_torch.weight.data.fill_(1.0)
        op_torch.bias.data.fill_(0.0)
        if mode == 'train':
            op_torch.train()
        else:
            op_torch.eval()
        lhs = op_torch(
            torch.from_numpy(x_torch).to(device_torch)).detach().cpu().numpy()
        assertAllClose(lhs,
                       nhwc2nchw(rhs),
                       rtol=1e-5,
                       atol=1e-8,
                       msg='{}/{}/{}/{}'.format(batch_norm_momentum,
                                                batch_norm_epsilon, fused,
                                                mode))

    def test_param(self):
        x = np.random.rand(3, 2, 1, 4).astype(np.float32)
        for batch_norm_momentum in [0.9, 0.99]:
            for batch_norm_epsilon in [1e-1, 1e-3, 1e-8]:
                for fused in [True, False]:
                    for mode in ['eval', 'train']:
                        self._test(x, batch_norm_momentum, batch_norm_epsilon,
                                   fused, mode)


if __name__ == "__main__":
    unittest.main()
