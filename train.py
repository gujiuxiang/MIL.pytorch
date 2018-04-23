# Use tensorboard

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time
from six.moves import cPickle
from dataloader import *
from model import *
import tensorflow as tf
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import opts
from model import eval_utils
from model import utils
from model import models

rusage_denom = 1024
printf = functools.partial(print, end="")

def extract_fts(opt, data):
    images = Variable(torch.from_numpy(data['images']), volatile=False).cuda()
    mil_label = Variable(torch.from_numpy(data['mil_label']),volatile=False).cuda()
    return images, mil_label

def record_training(opt, model, iteration, tf_summary_writer, current_record, history_record):
    [train_loss] = current_record
    [loss_history, lr_history] = history_record
    utils.add_summary_value(tf_summary_writer, 'train_lr', opt.learning_rate, iteration)
    utils.add_summary_value(tf_summary_writer, 'train_weight_decay', opt.weight_decay, iteration)
    utils.add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
    utils.add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
    tf_summary_writer.flush()
    loss_history[iteration] = train_loss
    lr_history[iteration] = opt.current_lr
    return history_record

def record_ckpt(opt, infos, model, optimizer, best_flag):
    tag = '-best' if best_flag else ''
    print("Save language model start")
    checkpoint_path = os.path.join(opt.checkpoint_path, opt.model_id + '.model' + tag + '.pth')
    optimizer_path = os.path.join(opt.checkpoint_path, opt.model_id + '.optimizer' + tag + '.pth')
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    print("Save infos start")
    with open(os.path.join(opt.checkpoint_path, opt.model_id + '.infos' + tag + '.pkl'), 'wb') as f:
        cPickle.dump(infos, f)
    print("model saved to {}".format(checkpoint_path))

def train(opt):
    print("Load dataset with image features, and labels\n")
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tf_summary_writer = tf.summary.FileWriter(opt.checkpoint_path)
    print("Load informations from infos.pkl ... ")
    opt, infos, iteration, epoch, val_history, train_history = utils.history_infos(opt)
    [loss_history, lr_history] = train_history
    [val_result_history, best_val_score, val_loss] = val_history

    # Update dataloader info
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)

    print("Build image cnn model, and initialize it with pre-trained cnn model")
    model, crit, optimizer, infos = models.build_models(opt, infos)

    update_lr_flag = True
    while True:
        gc.collect()  # collect cpu memory
        if update_lr_flag:
            utils.paramsForEpoch(opt, epoch, optimizer)
            utils.update_lr(opt, epoch, optimizer)  # Assign the learning rate

        data = loader.get_batch('train')  # Load data from train split (0)
        torch.cuda.synchronize()
        start = time.time()

        images, mil_label = extract_fts(opt, data)
        optimizer.zero_grad()
        crit_outputs = crit(model(images), mil_label)
        loss = crit_outputs[0]
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        train_loss = loss.data[0]

        last_name = os.path.basename(opt.model_id)
        last_time = last_name[0:8]
        print(
            "{}/{},{}/{},loss(t|{:.4f},v|{:.4f})|T/B({:.2f})" \
            .format(opt.model+'.'+last_time, iteration, epoch, opt.batch_size,
                    train_loss, val_loss,
                    time.time() - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            current_record = [train_loss]
            history_record = [loss_history, lr_history]
            history_record = record_training(opt, model, iteration, tf_summary_writer, current_record, history_record)
            [loss_history, lr_history] = history_record

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            eval_kwargs = {'split': 'test', 'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            eval_kwargs['split'] = 'test'
            eval_kwargs['dataset'] = opt.input_json
            val_loss = eval_utils.eval_split(opt, model, crit, loader, eval_kwargs)

            utils.add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
            tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss}

            # Save model if is improving on validation result
            current_score = val_loss
            best_flag = False
            if True:  # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['val_result_history'] = val_result_history
                infos['loss_history'] = loss_history
                infos['lr_history'] = lr_history
                infos['vocab'] = loader.get_vocab()
                # Dump checkpoint
                record_ckpt(opt, infos, model, optimizer, best_flag)
        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


'''
Main function: Start from here !!!
'''
opt = opts.parse_opt()
train(opt)
