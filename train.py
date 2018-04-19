import pdb
import argparse
import torch
import torch.optim as optim
from data_loader import *
from model import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_file', default='data/visdial_data.h5', help='path to visdial data hdf5 file')
    parser.add_argument('--img_file', default='data/data_img.h5', help='path to image hdf5 file')
    parser.add_argument('--training_epoch', default='100', help='training epoch')
    parser.add_argument('--batch_size', default='128', help='batch size')
    parser.add_argument('--lr', default='0.0001', help='initial learning rate')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--model_path', default='', help='folder to output model checkpoints')

    opt = parser.parse_args()
    print(opt)
    
    trainloader = get_loader(opt.data_file, opt.img_file, train=True)
    net = Encoder()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    
    for epoch in range(opt.training_epoch):
        # Train
        train_loss = 0
        net.train()
        for i, data in enumerate(trainloader):
            image, caption, question, answer, option, answerIdx = data
            optimizer.zero_grad()
            loss = net(image, caption, question, answer, option, answerIdx)
            loss.backward()
            nn.utils.clip_grad_norm(net.parameters(), 0.25, 2)
            optimizer.step()
            train_loss += loss.cpu().data.numpy()[0] / opt.batch_size
            if i % 100 == 0 and i > 0:
                print('Training loss: ', train_loss / 100)
                train_loss = 0

        print('Saving model...')
        torch.save(net, opt.model_path + 'torch_model_' + str(epoch) + '.npy')

        print('Learning rate: ', opt.learning_rate)
        if epoch % 20 == 19:
            opt.lr *= 0.5
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    print('Finished Training')
