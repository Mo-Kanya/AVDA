import time

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#return MNIST dataloader
def get_dataloader(data_dir, batch_size, train=True):
    if train:
        imgs = datasets.ImageFolder(data_dir,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.RandomCrop(224),
                                        transforms.ToTensor(),
                                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ]))
        data_loader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    else:
        imgs = datasets.ImageFolder(data_dir,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ]))
        data_loader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, num_workers=4, shuffle=False)
    return data_loader


def get_support(tar_dir, n):
    dataset=datasets.ImageFolder(tar_dir,
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ]))
    
    X,Y=[],[]
    classes=31*[n]
    ids = list(range(len(dataset)))
    np.random.shuffle(ids)
    for i in ids:
        if len(X)==n*31:
            break
        x,y=dataset[i]
        if classes[y]>0:
            X.append(x)
            Y.append(y)
            classes[y]-=1
    assert (len(X)==n*31)
    return torch.stack(X,dim=0),torch.from_numpy(np.array(Y))



def sample_data():
    repetition = 0
    
    src_dir = "./domain_adaptation_images/webcam/images/"
    train_set = datasets.ImageFolder(src_dir,
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ]))
    sample_per_class = 3
    n=len(train_set)
    X=torch.Tensor(n,3,224,224)
    Y=torch.LongTensor(n)

    inds=torch.randperm(n)
    for i,index in enumerate(inds):
        x,y=train_set[index]
        X[i]=x
        Y[i]=y
    return X,Y

"""
G1: a pair of pic comes from same domain ,same class
G3: a pair of pic comes from same domain, different classes

G2: a pair of pic comes from different domain,same class
G4: a pair of pic comes from different domain, different classes
"""
def create_groups(X_s,Y_s,X_t,Y_t,seed=1):
    #change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)


    n=X_t.shape[0] #10*shot


    #shuffle order
    classes = torch.unique(Y_t)
    classes=classes[torch.randperm(len(classes))]


    class_num=classes.shape[0]
    shot=n//class_num



    def s_idxs(c):
        idx=torch.nonzero(Y_s.eq(int(c)))

        return idx[torch.randperm(len(idx))][:shot*2].squeeze()
    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))


    source_matrix=torch.stack(source_idxs)

    target_matrix=torch.stack(target_idxs)


    G1, G2, G3, G4 = [], [] , [] , []
    Y1, Y2 , Y3 , Y4 = [], [] ,[] ,[]


    for i in range(10):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j*2]],X_s[source_matrix[i][j*2+1]]))
            Y1.append((Y_s[source_matrix[i][j*2]],Y_s[source_matrix[i][j*2+1]]))
            G2.append((X_s[source_matrix[i][j]],X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]],Y_t[target_matrix[i][j]]))
            G3.append((X_s[source_matrix[i%10][j]],X_s[source_matrix[(i+1)%10][j]]))
            Y3.append((Y_s[source_matrix[i % 10][j]], Y_s[source_matrix[(i + 1) % 10][j]]))
            G4.append((X_s[source_matrix[i%10][j]],X_t[target_matrix[(i+1)%10][j]]))
            Y4.append((Y_s[source_matrix[i % 10][j]], Y_t[target_matrix[(i + 1) % 10][j]]))



    groups=[G1,G2,G3,G4]
    groups_y=[Y1,Y2,Y3,Y4]


    return groups,groups_y




def sample_groups(X_s,Y_s,X_t,Y_t,seed=1):


    print("Sampling groups")
    return create_groups(X_s,Y_s,X_t,Y_t,seed=seed)


