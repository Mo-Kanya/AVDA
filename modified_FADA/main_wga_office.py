import argparse
import torch
import dataloader_off31 as dataloader
from models import main_models
# from config import config
from models import BasicModule
from utils.utils import *
from torch.utils.data import DataLoader
import numpy as np
import yaml

with open('./config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.Loader)

max_acu = 0.0
parser=argparse.ArgumentParser()
parser.add_argument('--n_epoches_1',type=int,default=config['n_epoches_1'])
parser.add_argument('--n_epoches_2',type=int,default=config['n_epoches_2'])
parser.add_argument('--n_epoches_3',type=int,default=config['n_epoches_3'])
parser.add_argument('--batch_size1',type=int,default=config['batch_size1'])
parser.add_argument('--batch_size2',type=int,default=config['batch_size2'])
parser.add_argument('--lr',type=float,default=config['lr'])
parser.add_argument('--batch_size',type=int,default=config['batch_size'])
parser.add_argument('--n_target_samples',type=int,default=config['n_target_samples'])
parser.add_argument('--seed',type=int,default=config['seed'])
opt=vars(parser.parse_args())

use_cuda=True if torch.cuda.is_available() else False
device=torch.device(f"cuda:{config['device']}") if use_cuda else torch.device('cpu')
torch.manual_seed(opt['seed'])
if use_cuda:
    torch.cuda.manual_seed(opt['seed'])

experiment_number = config['experiment_number']
batch_size = config['batch_size']
task = config['task']
src_dir, tar_dir = get_domains(task)
# domains = ["./domain_adaptation_images/webcam/images/",
#            "./domain_adaptation_images/dslr/images/",
#            "./domain_adaptation_images/amazon/images/"]
# src_dir = domains[1]
# tar_dir = domains[2]
test_dataloader = dataloader.get_dataloader(tar_dir, batch_size=batch_size, train=False)
train_set = dataloader.get_dataloader(src_dir, batch_size=batch_size, train=True)


classifier=main_models.Classifier()
encoder=main_models.Encoder()
discriminator=main_models.DCD(input_features=config['DCD']['input_features'],
                              h1_features=config['DCD']['h1_features'],
                              h2_features=config['DCD']['h2_features'])
attention = main_models.Attention(input_features=config['Attention']['input_features'],
                                  h_features=config['Attention']['h_features'],
                                  normalize=config['Attention']['normalize'])
# TODO: attention需要有初始化参数，感觉在0.5左右会好一点

classifier.to(device)
encoder.to(device)
discriminator.to(device)
attention.to(device)
loss_fn=torch.nn.CrossEntropyLoss()
X_s,Y_s=dataloader.sample_data()
X_t,Y_t=dataloader.get_support(tar_dir, 3)



# optimizer_all=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters())+list(attention.parameters())+list(discriminator.parameters()),lr=opt['lr'])
# optimizer_all=torch.optim.Adadelta(list(encoder.parameters())+list(classifier.parameters()))
#
# for epoch in range(opt['n_epoches_1']):
#
#     for data,labels in train_set:
#         data=data.to(device)
#         labels=labels.to(device)
#         optimizer_all.zero_grad()
#
#         encode = encoder(data)
#         # attention_score = attention(encode)
#         # y_pred=classifier(encode * (1-attention_score))
#         y_pred=classifier(encode)
#         loss=loss_fn(y_pred,labels)
#         loss.backward()
#
#         optimizer_all.step()
#
#     print(loss)
#
#     acc=0
#     deno = 0
#     with torch.no_grad():
#         for data,labels in test_dataloader:
#             data=data.to(device)
#             labels=labels.to(device)
#             encode = encoder(data)
#             # attention_score = attention(encode)
#             # y_test_pred=classifier(encode * (1-attention_score))
#             y_test_pred=classifier(encode)
#             # acc+=(torch.max(y_test_pred,1)[1]==labels).float().mean().item()
#             y_test_pred = torch.argmax(y_test_pred, dim=1)
#             acc += torch.sum(y_test_pred == labels).item()
#             deno += len(labels)
#
#         # accuracy=round(acc / float(len(test_dataloader)), 3)
#         accuracy=acc / deno
#
#         print("step1----Epoch %d/%d  accuracy: %.3f "%(epoch+1,opt['n_epoches_1'],accuracy))
#
# exit()

#-------------------training for step 3-------------------
optimizer_all=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters())+list(attention.parameters())+list(discriminator.parameters()), lr=config['lr'])
# test_dataloader=DataLoader(test_set,batch_size=opt['batch_size'],shuffle=True,drop_last=True)

as_record = []

for epoch in range(opt['n_epoches_3']):
    #---training g and h , DCD is frozen

    groups, groups_y = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=opt['n_epoches_2']+epoch)
    G1, G2, G3, G4 = groups
    Y1, Y2, Y3, Y4 = groups_y
    groups_2 = [G2, G4]
    groups_y_2 = [Y2, Y4]

    n_iters = 2 * len(G2)
    index_list = torch.randperm(n_iters)

    n_iters_dcd = 4 * len(G2)
    index_list_dcd = torch.randperm(n_iters_dcd)

    mini_batch_size_g_h = opt['batch_size1'] #data only contains G2 and G4 ,so decrease mini_batch
    mini_batch_size_dcd= opt['batch_size2'] #data contains G1,G2,G3,G4 so use 40 as mini_batch
    X1 = []
    X2 = []
    ground_truths_y1 = []
    ground_truths_y2 = []
    dcd_labels=[]

    discriminator.train()

    for index in range(n_iters):

        ground_truth=index_list[index]//len(G2)
        x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        # y1=torch.LongTensor([y1.item()])
        # y2=torch.LongTensor([y2.item()])
        dcd_label=0 if ground_truth==0 else 2
        X1.append(x1)
        X2.append(x2)
        ground_truths_y1.append(y1)
        ground_truths_y2.append(y2)
        dcd_labels.append(dcd_label)

        if (index+1)%mini_batch_size_g_h==0:

            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths_y1=torch.LongTensor(ground_truths_y1)
            ground_truths_y2 = torch.LongTensor(ground_truths_y2)
            dcd_labels=torch.LongTensor(dcd_labels)
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths_y1=ground_truths_y1.to(device)
            ground_truths_y2 = ground_truths_y2.to(device)
            dcd_labels=dcd_labels.to(device)

            optimizer_all.zero_grad()

            encoder_X1=encoder(X1)
            encoder_X2=encoder(X2)

            attention_score1 = attention(encoder_X1)
            attention_score2 = attention(encoder_X2)

            attention_X1 = encoder_X1 * attention_score1
            attention_X2 = encoder_X2 * attention_score2
            X_cat = torch.cat([attention_X1, attention_X2], 1)
            attention_X1_clf = encoder_X1 * (1-attention_score1)
            attention_X2_clf = encoder_X2 * (1-attention_score2)
            y_pred_X1 = classifier(attention_X1_clf)
            y_pred_X2 = classifier(attention_X2_clf)
            y_pred_dcd=discriminator(X_cat)

            loss_X1=loss_fn(y_pred_X1,ground_truths_y1)
            loss_X2=loss_fn(y_pred_X2,ground_truths_y2)
            loss_dcd=loss_fn(y_pred_dcd,dcd_labels)

            loss_sum = loss_X1 + loss_X2 + config['lambda'] * loss_dcd

            loss_sum.backward()
            optimizer_all.step()

            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []
            dcd_labels = []

    print('loss:', loss_sum.cpu().item())
    acc = 0
    deno = 0
    discriminator.eval()
    with torch.no_grad():
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)

            # y_test_pred = classifier(encoder(data))
            encoder_test = encoder(data)
            attention_score = attention(encoder_test)

            as_record.append(attention_score)

            attention_clf = encoder_test * (1 - attention_score)
            y_test_pred = classifier(attention_clf)
            y_test_pred = torch.argmax(y_test_pred, dim=1)
            acc += torch.sum(y_test_pred == labels).item()
            deno += len(labels)
            # acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

        # accuracy = round(acc / float(len(test_dataloader)), 3)
        accuracy = acc / deno

        print("step3----Epoch %d/%d  accuracy: %.3f " %
              (epoch + 1, opt['n_epoches_3'], accuracy))
        if (accuracy>max_acu):
            max_acu = accuracy
print("Max Accuracy: %f" % max_acu)
print(list(attention.parameters()))

as_record = torch.cat(as_record).cpu().numpy()

with open('as_record.npy', 'wb') as f:
    np.save(f, as_record, allow_pickle=True)

# save
f_name = f'wga_office_{task}'
if config['save_model']:
    mkdir(f'./result/{f_name}/result_{experiment_number}')
    torch.save(encoder.state_dict(), f'./result/{f_name}/result_{experiment_number}/encoder.pth')
    torch.save(attention.state_dict(), f'./result/{f_name}/result_{experiment_number}/attention.pth')
    torch.save(classifier.state_dict(), f'./result/{f_name}/result_{experiment_number}/classifier.pth')
    torch.save(discriminator.state_dict(), f'./result/{f_name}/result_{experiment_number}/discriminator.pth')

# save config
exe_cmd(f'cp ./config/config.yaml ./result/{f_name}/result_{experiment_number}/config.yaml')


