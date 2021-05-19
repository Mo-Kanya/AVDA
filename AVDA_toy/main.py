import argparse
import torch
import dataloader
from models import main_models
import numpy as np
from matplotlib import pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--lambda1',type=float,default=2.5)
parser.add_argument('--lambda2',type=float,default=1.5)
parser.add_argument('--n_epoches_1',type=int,default=200)
parser.add_argument('--n_epoches_2',type=int,default=100)
parser.add_argument('--n_epoches_3',type=int,default=200)
parser.add_argument('--n_target_samples',type=int,default=7)
parser.add_argument('--batch_size',type=int,default=64)

opt=vars(parser.parse_args())
lambda1 = opt['lambda1']
lambda2 = opt['lambda2']

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

print("\n---------------------------------------\n")
"""Pre-train"""
X_s,Y_s=dataloader.sample_data()
X_t,Y_t=dataloader.create_target_samples(opt['n_target_samples'])
train_dataloader=dataloader.mnist_dataloader(batch_size=40,train=True)
train_iter = iter(train_dataloader)
test_dataloader=dataloader.mnist_dataloader(batch_size=opt['batch_size'],train=False)

classifier=main_models.Classifier()
encoder=main_models.Encoder()
discriminator=main_models.DCD()
classifier.to(device)
encoder.to(device)
discriminator.to(device)

loss_fn=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters())+list(discriminator.parameters()), lr=0.002)
for epoch in range(opt['n_epoches_1']):
    # data
    groups,aa = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=epoch)

    n_iters = 4 * len(groups[1])
    index_list = torch.randperm(n_iters)
    mini_batch_size=40
    loss_mean = []
    X1 = [];X2 = [];ground_truths = []
    for index in range(n_iters):

        ground_truth=index_list[index]//len(groups[1])
        x1,x2=groups[ground_truth][index_list[index]-len(groups[1])*ground_truth]
        X1.append(x1)
        X2.append(x2)
        ground_truths.append(ground_truth)

        if (index+1)%mini_batch_size==0:
            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths=torch.LongTensor(ground_truths)
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths=ground_truths.to(device)

            data, labels = next(train_iter)
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            X_cat=torch.cat([encoder(X1),encoder(X2)],1)
            y_pred1=discriminator(X_cat.detach())
            loss1=loss_fn(y_pred1,ground_truths)

            y_pred2 = classifier(encoder(data))
            loss2 = loss_fn(y_pred2, labels)

            loss = lambda1*loss1 + loss2
            loss.backward()
            optimizer.step()
            loss_mean.append(loss1.item())
            X1 = []
            X2 = []
            ground_truths = []
    if epoch%10==0:
        print("step2----Epoch %d/%d loss:%.3f"%(epoch+1,opt['n_epoches_2'],np.mean(loss_mean)))

acc=0
for data,labels in test_dataloader:
    data=data.to(device)
    labels=labels.to(device)
    y_test_pred=classifier(encoder(data))
    acc+=(torch.max(y_test_pred,1)[1]==labels).float().mean().item()
accuracy=round(acc / float(len(test_dataloader)), 3)
print("Classification Accuracy after pretraining: %.3f "%(accuracy))


"""Train attention"""
test_dataloader=dataloader.svhn_dataloader(train=False,batch_size=opt['batch_size'])
X_s,Y_s=dataloader.sample_data()
X_t,Y_t=dataloader.create_target_samples(opt['n_target_samples'])

attention = main_models.Attention()
attention.to(device)
# torch.nn.init.constant_(attention.fc1.weight, 0)

optimizer_h = torch.optim.Adam(classifier.parameters(), lr=0.002)
optimizer_a = torch.optim.Adam(attention.parameters(), lr=0.001)
optimizer_d = torch.optim.Adam(discriminator.parameters(),lr=0.002)

for epoch in range(opt['n_epoches_3']):

    groups, groups_y = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=opt['n_epoches_2']+epoch)
    G1, G2, G3, G4 = groups
    Y1, Y2, Y3, Y4 = groups_y
    groups_2 = [G2, G4]
    groups_y_2 = [Y2, Y4]
    n_iters = 2 * len(G2)
    index_list = torch.randperm(n_iters)
    n_iters_dcd = 4 * len(G2)
    index_list_dcd = torch.randperm(n_iters_dcd)
    n_iters_a = len(G2)
    mini_batch_size_g_h = 20 # data only contains G2 and G4 ,so decrease mini_batch
    mini_batch_size_dcd= 40 # data contains G1,G2,G3,G4 so use 40 as mini_batch

    # testing
    if epoch%2 == 0:
        acc = 0
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                a_score = attention(torch.FloatTensor([1]).to(device))
                y_test_pred = classifier(encoder(data) * (1-a_score))
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
        accuracy = round(acc / float(len(test_dataloader)), 3)
        print("Training A----Epoch %d/%d  accuracy: %.3f " %(epoch + 1, opt['n_epoches_3'], accuracy))

    # update A
    X1 = []
    X2 = []
    ground_truths_y1 = []
    ground_truths_y2 = []
    dcd_labels=[]
    for index in range(n_iters_a):
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

            optimizer_a.zero_grad()
            optimizer_h.zero_grad()

            with torch.no_grad():
                encoder_X1=encoder(X1)
                encoder_X2=encoder(X2)

            attention_score1 = attention(torch.FloatTensor([1]).to(device))
            attention_score2 = attention(torch.FloatTensor([1]).to(device))

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

            loss_sum = loss_X1 + loss_X2 + loss_dcd*lambda2

            loss_sum.backward()
            # optimizer_h.step()
            optimizer_a.step()

            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []
            dcd_labels = []

    # train H
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

            optimizer_a.zero_grad()
            optimizer_h.zero_grad()

            with torch.no_grad():
                encoder_X1=encoder(X1)
                encoder_X2=encoder(X2)
                attention_score1 = attention(torch.FloatTensor([1]).to(device))
                attention_score2 = attention(torch.FloatTensor([1]).to(device))

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

            loss_sum = loss_X1 + loss_X2

            loss_sum.backward()
            optimizer_h.step()
            # optimizer_a.step()

            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []
            dcd_labels = []

    # train DCD
    X1 = []
    X2 = []
    ground_truths = []
    for index in range(n_iters_dcd):

        ground_truth=index_list_dcd[index]//len(groups[1])

        x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
        X1.append(x1)
        X2.append(x2)
        ground_truths.append(ground_truth)

        if (index + 1) % mini_batch_size_dcd == 0:
            X1 = torch.stack(X1)
            X2 = torch.stack(X2)
            ground_truths = torch.LongTensor(ground_truths)
            X1 = X1.to(device)
            X2 = X2.to(device)
            ground_truths = ground_truths.to(device)

            optimizer_d.zero_grad()

            encoder_X1 = encoder(X1)
            encoder_X2 = encoder(X2)

            attention_score1 = attention(torch.FloatTensor([1]).to(device))
            attention_score2 = attention(torch.FloatTensor([1]).to(device))

            attention_X1 = encoder_X1 * attention_score1
            attention_X2 = encoder_X2 * attention_score2

            X_cat = torch.cat([attention_X1, attention_X2], 1)
            y_pred_X1 = classifier(attention_X1)
            y_pred_X2 = classifier(attention_X2)
            y_pred_dcd = discriminator(X_cat.detach())

            # y_pred = discriminator(X_cat.detach())
            loss = loss_fn(y_pred_dcd, ground_truths)
            loss.backward()
            optimizer_d.step()
            # loss_mean.append(loss.item())
            X1 = []
            X2 = []
            ground_truths = []

print("attention score after train:")
print(attention.fc1.weight)





