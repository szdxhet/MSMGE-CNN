import torch
import os
import sys
import argparse
import time
import numpy as np
from tensorboardX import SummaryWriter
from braindecode.datautil.iterators import BalancedBatchSizeIterator

from tools.utils import set_seed, set_save_path, Logger, save, load_adj, EarlyStopping
from tools.run_tools import train_one_epoch_classifier, evaluate_one_epoch_classifier
from models.multi_scale_multi_graph_nets import MSMGECNN
from tools.data_bciciv2a_tools import load_bciciv2a_data_single_subject


def run(args):
    # ----------------------------------------------environment setting-----------------------------------------------
    set_seed(args.seed)
    args = set_save_path(args.father_path, args)
    sys.stdout = Logger(os.path.join(args.log_path, 'information.txt'))
    tensorboard = SummaryWriter(args.tensorboard_path)
    start_epoch = 0

    # ------------------------------------------------device setting--------------------------------------------------
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ------------------------------------------------data setting----------------------------------------------------
    if 'bci2a'==args.dataset:
        train_X, train_y, test_X, test_y = load_bciciv2a_data_single_subject(args.data_path,subject_id=args.id)
   
    iterator = BalancedBatchSizeIterator(batch_size=args.batch_size)

    # -----------------------------------------------training setting-------------------------------------------------

    
    if 'bci2a' == args.dataset:
        Adj1 = torch.tensor(load_adj('bciciv2a'), dtype=torch.float32)
        train_data = train_X.permute(0, 2, 1).contiguous().reshape(-1, 22)
        Adj2= torch.tensor(np.corrcoef(train_data.numpy().T, ddof=1), dtype=torch.float32)
        model_classifier = MSMGECNN(Adj1, Adj2, 22, 4, k=args.k, input_time_length=1125, Adj_learn=args.adj_learn,
                                  drop_prob=args.dropout, pool_mode=args.pool, f1=8, f2=16, kernel_length=64)

    opt_classifier = torch.optim.Adam(model_classifier.parameters(), lr=args.lr, weight_decay=args.w_decay)
    criterion = torch.nn.CrossEntropyLoss()
    stop_train = EarlyStopping(patience=160, max_epochs=args.epochs)

    # -----------------------------------------------resume setting--------------------------------------------------
    best_acc = 0
    if args.resume_path is not None:
        load_checkpoints = torch.load(args.resume_path)
        model_classifier.load_state_dict(load_checkpoints['model_classifier'])
        start_epoch = load_checkpoints['epoch'] - 1
        best_acc = load_checkpoints['acc']

    # -------------------------------------------------run------------------------------------------------------------

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if stop_train.early_stop:
            print("early stop in {}!".format(epoch))
            break
        train_one_epoch_classifier(epoch, iterator, (train_X, train_y), model_classifier, device, opt_classifier,
                                   criterion, tensorboard, start_time, args)
        avg_acc, avg_loss= evaluate_one_epoch_classifier(epoch, iterator, (test_X, test_y), model_classifier, device,
                                                          criterion, tensorboard, args, start_time)

        stop_train(avg_acc)
        save_checkpoints = {'model_classifier': model_classifier.state_dict(),
                            'epoch': epoch + 1,
                            'acc': avg_acc}


        if avg_acc > best_acc:
            best_acc = avg_acc
            save(save_checkpoints, os.path.join(args.model_classifier_path, 'model_classifier_best.pth.tar'))
        print('best_acc:{}'.format(best_acc))
        save(save_checkpoints, os.path.join(args.model_classifier_path, 'model_classifier_newest.pth.tar'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str,
                        default='',
                        help='The father path of pkl file')
    parser.add_argument('-dataset', type=str, default='bci2a',
                        help='Different datasets correspond to different data processing methods.')

    parser.add_argument('-id', type=int, default=9, help='Subject id used to train and test.')
    parser.add_argument('-adj_learn',  default=True,action='store_false', help='Ajd is trainable')
    parser.add_argument('-k', type=int, default=1, help='The order of graph embedding')
    parser.add_argument('-pool', type=str, default='mean', choices=['max', 'mean'])
    parser.add_argument('-dropout', type=float, default=0.2, help='Dropout rate.')

    parser.add_argument('-epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('-batch_size', default=32, type=int, help='Batch size.')
    parser.add_argument('-lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('-w_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')

    parser.add_argument('-father_path', type=str, default='save_signle', help='The father path of log files.')
    parser.add_argument('-print_freq', type=int, default=16, help='The frequency to show training information.')
    parser.add_argument('-seed', type=int, default='2', help='Random seed.')
    parser.add_argument('-resume_path', type=str, default=None, help='Path of saved model.')
    args_ = parser.parse_args()
    run(args_)





    

