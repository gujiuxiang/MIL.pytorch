import argparse
import datetime
import os


def parse_opt():
    parser = argparse.ArgumentParser()
    # RL setting
    parser.add_argument('--model', type=str, default='vgg19')
    parser.add_argument('--learning', type=str, default='mle')
    parser.add_argument('--start_from', type=str, default='')
    # Data input settings
    parser.add_argument('--fc_feat_size', type=int, default=2048)  # '2048 for resnet, 4096 for vgg'
    parser.add_argument('--att_feat_size', type=int, default=2048)  # '2048 for resnet, 512 for vgg'
    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1)  # 'number of epochs'
    parser.add_argument('--batch_size', type=int, default=2)  # 'minibatch size'
    parser.add_argument('--seq_per_img', type=int,default=5)  # number of captions to sample for each image during training.
    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam')  # rmsprop|sgd|sgdmom|adagrad|adam
    parser.add_argument('--learning_rate', type=float, default=4e-4)  # 'learning rate'
    parser.add_argument('--learning_rate_decay_start', type=int,default=0)  # at what iteration to start decaying learning rate? (-1 = dont) (in epoch)
    parser.add_argument('--learning_rate_decay_every', type=int,default=5000)  # every how many iterations thereafter to drop LR?(in epoch)
    parser.add_argument('--learning_rate_decay_rate', type=float,default=0.8)  # every how many iterations thereafter to drop LR?(in epoch)
    parser.add_argument('--optim_alpha', type=float, default=0.8)  # alpha for adam
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--optim_beta', type=float, default=0.999)  # beta used for adam
    parser.add_argument('--optim_epsilon', type=float, default=1e-8)  # epsilon that goes into denominator for smoothing
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # weight_decay
    parser.add_argument('--grad_clip', type=float, default=0.1)  # clip gradients at this value
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)  # strength of dropout in the Language Model RNN
    # Datasets
    parser.add_argument('--input_json', type=str, default='data/mscoco/cocotalk_karpathy.json')
    parser.add_argument('--input_im_h5', type=str, default='data/mscoco/cocotalk_karpathy.h5')
    parser.add_argument('--input_label_h5', type=str, default='data/mscoco/cocotalk_karpathy_label_semantic_words.h5')
    # Evaluation/Checkpointing
    parser.add_argument('--split', type=str, default='train')  # Dataset split type
    parser.add_argument('--val_images_use', type=int, default=5000)  # number of images for period validation (-1 = all)
    parser.add_argument('--save_checkpoint_every', type=int,default=100)
    parser.add_argument('--checkpoint_path', type=str, default='save')  # directory to store checkpointed models'
    parser.add_argument('--losses_log_every', type=int, default=25)  # How often do we snapshot losses, (0 = disable)
    parser.add_argument('--load_best_score', type=int, default=1)  # load previous best score when resuming training.
    # misc
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--train_only', type=int, default=0)  # If true then use 80k, else use 110k
    parser.add_argument('--gpus', default=[0, 1], nargs='+', type=int)  # Use CUDA on the listed devices
    parser.add_argument('--model_id', type=str, default='')  # Id identifying this run/job.
    # used in cross-val and appended when writing progress files'

    args = parser.parse_args()
    # Check if args are valid
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.load_best_score == 0 or args.load_best_score == 1, "should be 0 or 1"

    # Update args
    args.gpus = range(args.n_gpus)
    last_name = os.path.basename(args.start_from)
    last_time = last_name[0:8]
    if len(args.start_from):
        args.model_id = last_name
    else:
        args.model_id = datetime.datetime.now().strftime("%m%d%H%M") + "_mil_" + args.model + '_' + args.learning
    args.checkpoint_path = args.checkpoint_path + '/' + args.model_id
    return args
