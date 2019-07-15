"""Train and val."""
import logging
import os
import time

import torch

from utils.config import FLAGS, _ENV_EXPAND
from utils.common import get_params_by_name
from utils.common import set_random_seed
from utils.common import create_exp_dir
from utils.common import setup_logging
from utils.common import save_status
from utils.common import get_device
from utils.common import extract_item
from utils.common import get_data_queue_size
from utils.common import bn_calibration
from utils import dataflow
from utils import optim
from utils import distributed as udist

import common as mc


def run_one_epoch(epoch,
                  loader,
                  model,
                  criterion,
                  optimizer,
                  lr_scheduler,
                  ema,
                  meters,
                  max_iter=None,
                  phase='train'):
    """Run one epoch."""
    assert phase in ['train', 'val', 'test', 'bn_calibration'
                    ], "phase not be in train/val/test/bn_calibration."
    train = phase == 'train'
    if train:
        model.train()
    else:
        model.eval()
    if phase == 'bn_calibration':
        model.apply(bn_calibration)

    if FLAGS.use_distributed:
        loader.sampler.set_epoch(epoch)

    results = None
    data_iterator = iter(loader)
    if FLAGS.use_distributed:
        data_fetcher = dataflow.DataPrefetcher(data_iterator)
    else:
        # TODO(meijieru): prefetch for non distributed
        logging.warning('Not use prefetcher')
        data_fetcher = data_iterator
    for batch_idx, (input, target) in enumerate(data_fetcher):
        # used for bn calibration
        if max_iter is not None:
            assert phase == 'bn_calibration'
            if batch_idx >= max_iter:
                break

        target = target.cuda(non_blocking=True)
        if train:
            optimizer.zero_grad()
            loss = mc.forward_loss(model, criterion, input, target, meters)
            loss_l2 = optim.cal_l2_loss(model, FLAGS.weight_decay,
                                        FLAGS.weight_decay_method)
            loss = loss + loss_l2
            loss.backward()
            if FLAGS.use_distributed:
                udist.allreduce_grads(model)

            if FLAGS._global_step % FLAGS.log_interval == 0:
                results = mc.reduce_and_flush_meters(meters)
                if udist.is_master():
                    logging.info('Epoch {}/{} Iter {}/{} {}: '.format(
                        epoch, FLAGS.num_epochs, batch_idx, len(loader), phase)
                                 + ', '.join('{}: {:.4f}'.format(k, v)
                                             for k, v in results.items()))
                    for k, v in results.items():
                        mc.summary_writer.add_scalar('{}/{}'.format(phase, k),
                                                     v, FLAGS._global_step)
            if udist.is_master(
            ) and FLAGS._global_step % FLAGS.log_interval == 0:
                mc.summary_writer.add_scalar('train/learning_rate',
                                             optimizer.param_groups[0]['lr'],
                                             FLAGS._global_step)
                mc.summary_writer.add_scalar('train/l2_regularize_loss',
                                             extract_item(loss_l2),
                                             FLAGS._global_step)
                mc.summary_writer.add_scalar(
                    'train/current_epoch',
                    FLAGS._global_step / FLAGS._steps_per_epoch,
                    FLAGS._global_step)
                if FLAGS.data_loader_workers > 0:
                    mc.summary_writer.add_scalar(
                        'data/train/prefetch_size',
                        get_data_queue_size(data_iterator), FLAGS._global_step)

            optimizer.step()
            lr_scheduler.step()
            if FLAGS.use_distributed and FLAGS.allreduce_bn:
                udist.allreduce_bn(model)
            FLAGS._global_step += 1

            # NOTE: after steps count upate
            if ema is not None:
                model_unwrap = mc.unwrap_model(model)
                ema_names = ema.average_names()
                params = get_params_by_name(model_unwrap, ema_names)
                for name, param in zip(ema_names, params):
                    ema(name, param, FLAGS._global_step)
        else:
            mc.forward_loss(model, criterion, input, target, meters)

    if not train:
        results = mc.reduce_and_flush_meters(meters)
        if udist.is_master():
            logging.info(
                'Epoch {}/{} {}: '.format(epoch, FLAGS.num_epochs, phase)
                + ', '.join(
                    '{}: {:.4f}'.format(k, v) for k, v in results.items()))
            for k, v in results.items():
                mc.summary_writer.add_scalar('{}/{}'.format(phase, k), v,
                                             FLAGS._global_step)
    return results


def train_val_test():
    """Train and val."""
    torch.backends.cudnn.benchmark = True

    # model
    model, model_wrapper = mc.get_model()
    ema = mc.setup_ema(model)
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    criterion_smooth = optim.CrossEntropyLabelSmooth(
        FLAGS.model_kwparams['num_classes'],
        FLAGS['label_smoothing'],
        reduction='none').cuda()
    # TODO(meijieru): cal loss on all GPUs instead only `cuda:0` when non
    # distributed

    if FLAGS.get('log_graph_only', False):
        if udist.is_master():
            _input = torch.zeros(1, 3, FLAGS.image_size,
                                 FLAGS.image_size).cuda()
            _input = _input.requires_grad_(True)
            mc.summary_writer.add_graph(model_wrapper, (_input,), verbose=True)
        return

    # check pretrained
    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained,
                                map_location=lambda storage, loc: storage)
        if ema:
            ema.load_state_dict(checkpoint['ema'])
            ema.to(get_device(model))
        # update keys from external models
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if (hasattr(FLAGS, 'pretrained_model_remap_keys')
                and FLAGS.pretrained_model_remap_keys):
            new_checkpoint = {}
            new_keys = list(model_wrapper.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new, key_old in zip(new_keys, old_keys):
                new_checkpoint[key_new] = checkpoint[key_old]
                logging.info('remap {} to {}'.format(key_new, key_old))
            checkpoint = new_checkpoint
        model_wrapper.load_state_dict(checkpoint)
        logging.info('Loaded model {}.'.format(FLAGS.pretrained))
    optimizer = optim.get_optimizer(model_wrapper, FLAGS)

    # check resume training
    if FLAGS.resume:
        checkpoint = torch.load(os.path.join(FLAGS.resume,
                                             'latest_checkpoint.pt'),
                                map_location=lambda storage, loc: storage)
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if ema:
            ema.load_state_dict(checkpoint['ema'])
            ema.to(get_device(model))
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = optim.get_lr_scheduler(optimizer, FLAGS)
        lr_scheduler.last_epoch = (last_epoch + 1) * FLAGS._steps_per_epoch
        best_val = extract_item(checkpoint['best_val'])
        train_meters, val_meters = checkpoint['meters']
        FLAGS._global_step = (last_epoch + 1) * FLAGS._steps_per_epoch
        if udist.is_master():
            logging.info('Loaded checkpoint {} at epoch {}.'.format(
                FLAGS.resume, last_epoch))
    else:
        lr_scheduler = optim.get_lr_scheduler(optimizer, FLAGS)
        # last_epoch = lr_scheduler.last_epoch
        last_epoch = -1
        best_val = 1.
        train_meters = mc.get_meters('train')
        val_meters = mc.get_meters('val')
        FLAGS._global_step = 0

    if not FLAGS.resume and udist.is_master():
        logging.info(model_wrapper)
    if FLAGS.profiling:
        if 'gpu' in FLAGS.profiling:
            mc.profiling(model, use_cuda=True)
        if 'cpu' in FLAGS.profiling:
            mc.profiling(model, use_cuda=False)

    # data
    (train_transforms, val_transforms,
     test_transforms) = dataflow.data_transforms(FLAGS)
    (train_set, val_set, test_set) = dataflow.dataset(train_transforms,
                                                      val_transforms,
                                                      test_transforms, FLAGS)
    (train_loader, calib_loader, val_loader,
     test_loader) = dataflow.data_loader(train_set, val_set, test_set, FLAGS)

    if FLAGS.test_only and (test_loader is not None):
        if udist.is_master():
            logging.info('Start testing.')
        test_meters = mc.get_meters('test')
        validate(last_epoch, calib_loader, test_loader, criterion, test_meters,
                 model_wrapper, ema, 'test')
        return

    # already broadcast by AllReduceDistributedDataParallel
    # optimizer load same checkpoint/same initialization

    if udist.is_master():
        logging.info('Start training.')

    for epoch in range(last_epoch + 1, FLAGS.num_epochs):
        # train
        results = run_one_epoch(epoch,
                                train_loader,
                                model_wrapper,
                                criterion_smooth,
                                optimizer,
                                lr_scheduler,
                                ema,
                                train_meters,
                                phase='train')

        # val
        results = validate(epoch, calib_loader, val_loader, criterion,
                           val_meters, model_wrapper, ema, 'val')
        if results['top1_error'] < best_val:
            best_val = results['top1_error']

            if udist.is_master():
                save_status(model_wrapper, optimizer, ema, epoch, best_val,
                            (train_meters, val_meters),
                            os.path.join(FLAGS.log_dir, 'best_model.pt'))
                logging.info(
                    'New best validation top1 error: {:.4f}'.format(best_val))

        if udist.is_master():
            # save latest checkpoint
            save_status(model_wrapper, optimizer, ema, epoch, best_val,
                        (train_meters, val_meters),
                        os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'))

        # NOTE(meijieru): from scheduler code, should be called after train/val
        # use stepwise scheduler instead
        # lr_scheduler.step()
    return


def validate(epoch, calib_loader, val_loader, criterion, val_meters,
             model_wrapper, ema, phase):
    """Calibrate and validate."""
    assert phase in ['test', 'val']
    model_eval_wrapper = mc.get_ema_model(ema, model_wrapper)

    # bn_calibration
    if FLAGS.get('bn_calibration', False):
        if not FLAGS.use_distributed:
            logging.warning(
                'Only GPU0 is used when calibration when use DataParallel')
        with torch.no_grad():
            _ = run_one_epoch(epoch,
                              calib_loader,
                              model_eval_wrapper,
                              criterion,
                              None,
                              None,
                              None,
                              val_meters,
                              max_iter=FLAGS.bn_calibration_steps,
                              phase='bn_calibration')
        if FLAGS.use_distributed:
            udist.allreduce_bn(model_eval_wrapper)

    # val
    with torch.no_grad():
        results = run_one_epoch(epoch,
                                val_loader,
                                model_eval_wrapper,
                                criterion,
                                None,
                                None,
                                None,
                                val_meters,
                                phase=phase)
    return results


def main():
    """Entry."""
    NUM_IMAGENET_TRAIN = 1281167

    mc.setup_distributed(NUM_IMAGENET_TRAIN)
    if udist.is_master():
        FLAGS.log_dir = '{}/{}'.format(FLAGS.log_dir,
                                       time.strftime("%Y%m%d-%H%M%S"))
        # yapf: disable
        create_exp_dir(FLAGS.log_dir, FLAGS.config_path, blacklist_dirs=[
            'exp', '.git', 'pretrained', 'tmp', 'deprecated', 'bak'])
        # yapf: enable
        setup_logging(FLAGS.log_dir)
        for k, v in _ENV_EXPAND.items():
            logging.info('Env var expand: {} to {}'.format(k, v))
        logging.info(FLAGS)

    set_random_seed(FLAGS.get('random_seed', 0))
    with mc.SummaryWriterManager():
        train_val_test()


if __name__ == "__main__":
    main()
