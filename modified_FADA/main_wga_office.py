import argparse
import torch
import dataloader_off31 as dataloader
from models import main_models
from models import BasicModule
from torch.utils.data import DataLoader
import numpy as np
max_acu = 0.0
parser=argparse.ArgumentParser()
parser.add_argument('--n_epoches_1',type=int,default=10)
parser.add_argument('--n_epoches_2',type=int,default=100)
parser.add_argument('--n_epoches_3',type=int,default=150)
parser.add_argument('--batch_size1',type=int,default=20)
parser.add_argument('--batch_size2',type=int,default=40)
parser.add_argument('--lr',type=float,default=0.002)
parser.add_argument('--batch_size',type=int,default=16)
parser.add_argument('--n_target_samples',type=int,default=7)
parser.add_argument('--seed',type=int,default=17)
opt=vars(parser.parse_args())

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(opt['seed'])
if use_cuda:
    torch.cuda.manual_seed(opt['seed'])
domain_adaptation_task = 'MNIST_to_USPS' #'USPS_to_MNIST' 
# 'MNIST_to_USPS'
repetition = 0
sample_per_class = 7
batch_size = 16
tar_dir = "./domain_adaptation_images/dslr/images/"
src_dir = "./domain_adaptation_images/webcam/images/"
test_dataloader = dataloader.get_dataloader(src_dir, batch_size = batch_size, train=False)  # a fake one
train_set = dataloader.get_dataloader(src_dir, batch_size = batch_size, train=False)


classifier=main_models.Classifier()
encoder=main_models.Encoder()
discriminator=main_models.DCD(input_features=2048*2)
attention = main_models.Attention(input_features=2048)
# TODO: attention需要有初始化参数，感觉在0.5左右会好一点

classifier.to(device)
encoder.to(device)
discriminator.to(device)
attention.to(device)
loss_fn=torch.nn.CrossEntropyLoss()
X_s,Y_s=dataloader.sample_data()
X_t,Y_t=dataloader.get_support(tar_dir, 3)

#-------------------training for step 3-------------------
optimizer_all=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters())+list(attention.parameters())+list(discriminator.parameters()),lr=opt['lr'])
# test_dataloader=DataLoader(test_set,batch_size=opt['batch_size'],shuffle=True,drop_last=True)


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

            loss_sum = loss_X1 + loss_X2 + 0.2 * loss_dcd

            loss_sum.backward()
            optimizer_all.step()

            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []
            dcd_labels = []
    acc = 0
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)

        # y_test_pred = classifier(encoder(data))
        encoder_test = encoder(data)
        attention_score = attention(encoder_test)
        attention_clf = encoder_test * (1 - attention_score)
        y_test_pred = classifier(attention_clf)

        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

    accuracy = round(acc / float(len(test_dataloader)), 3)

    print("step3----Epoch %d/%d  accuracy: %.3f " %
          (epoch + 1, opt['n_epoches_3'], accuracy))
    if (accuracy>max_acu):
        max_acu = accuracy
print("Max Accuracy: %f" % max_acu)
print(list(attention.parameters()))
