import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from model import Model
from loader import data_loader
import sys
import argparse
import json



def save_model(tbs,path_num,ep,args):
    if not os.path.exists('save/model_{}'.format(path_num)):
        os.system('mkdir save/model_{}'.format(path_num))
        with open('save/model_{}/modeldescription'.format(path_num), 'w') as f:
            f.write('model description\n')
            f.write('batch size : {}\n'.format(args.b_size))
            f.write('hidden size : {}\n'.format(args.hidden))
            f.write('optimizer : {}\n'.format(args.optim))
            f.write('learning rate : {}\n'.format(args.lr))
    torch.save(tbs, 'save/model_{}/epoch{:02}'.format(path_num, ep + 1))



def train(args):
    torch.cuda.manual_seed(1000)
    print('Loading data')

    if args.imgf_path ==None:
        #default bottom-up top-down
        loader = data_loader(b_size=args.b_size)
    else:
        loader = data_loader(b_size=args.b_size,image_path=args.imgf_path)

    model = Model(v_size=loader.v_size,
                  K=loader.K,
                  f_dim=loader.f_dim,
                  h_dim=args.hidden,
                  o_dim=loader.o_dim,
                  pretrained_we=loader.we_matrix)

    criterion = nn.BCEWithLogitsLoss()

    # Move it to GPU
    model = model.cuda()
    criterion = criterion.cuda()

    if args.optim =='adam':
        optim = torch.optim.Adam(model.parameters(),lr=args.lr)
    elif args.optim =='adadelta':
        optim = torch.optim.Adadelta(model.parameters(),lr=args.lr)
    elif args.optim == 'adagrad':
        optim = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        sys.exit('Invalid optimizer.(adam, adadelta, adagrad, SGD)')


    # Continue training from saved model
    if args.savedmodel and os.path.isfile(args.savedmodel):
        print('Reading Saved model {}'.format(args.savedmodel))
        ckpt = torch.load(args.savedmodel)
        model.load_state_dict(ckpt['state_dict'])
        optim.load_state_dict(ckpt['optimizer'])

    # Training script
    print('Start training.')

    # for model save
    path_num=1
    while os.path.exists('save/model_{}'.format(path_num)):
        path_num+=1


    for ep in range(args.epoch):
        ep_loss = 0
        ep_correct = 0

        for step in range(loader.n_batch):
            # Batch preparation
            q_batch, a_batch, i_batch = loader.next_batch()
            q_batch = Variable(torch.from_numpy(q_batch))
            a_batch = Variable(torch.from_numpy(a_batch))
            i_batch = Variable(torch.from_numpy(i_batch))
            q_batch, a_batch, i_batch = q_batch.cuda(), a_batch.cuda(), i_batch.cuda()

            # Do model forward
            output = model(q_batch, i_batch)
            # print(output.shape)
            loss = criterion(output, a_batch)

            if step % 400 == 0 :
                _, oix = output.data.max(1)
                _, aix = a_batch.data.max(1)

                correct = torch.eq(oix, aix).sum()
                ep_correct += correct
                ep_loss += loss.data[0]
                print('Epoch %02d(%03d/%03d), loss: %.3f, correct: %3d / %d (%.2f%%)' %
                      (ep + 1, step, loader.n_batch, loss.data[0], correct, args.b_size, correct * 100 / args.b_size))

            # compute gradient and do optim step
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Save model after every epoch
        tbs = {
            'epoch': ep + 1,
            'loss': ep_loss / loader.n_batch,
            'accuracy': ep_correct * 100 / (loader.n_batch * args.b_size),
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict()
        }
        save_model(tbs, path_num, ep, args)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='VQA CVPR 17 SOTA Pytorch 3.6 ')
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--epoch',default=50,type=int)
    parser.add_argument('--b_size',default=512,type=int)
    parser.add_argument('--hidden',default=512,type=int)
    parser.add_argument('--lr',default=0.001,type=int)
    parser.add_argument('--optim',default='adam')
    parser.add_argument('--savedmodel',type=str,help='saved model path')
    parser.add_argument('--imgf_path',type=str,help='saved image feature path',default= None)
    args, unparsed = parser.parse_known_args()


    train(args)