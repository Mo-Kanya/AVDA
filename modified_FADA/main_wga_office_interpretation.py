import argparse
import torch
import dataloader_off31 as dataloader
from models import main_models
import numpy as np
import pickle
import yaml
from utils.utils import *

if __name__ == "__main__":
    pass

task = 'd2w'
experiment_number = 2

with open(f'./result/wga_office_{task}/result_{experiment_number}/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.Loader)

config['n_epoches_3'] = 10

batch_size = config['batch_size']
domains = ["webcam","dslr","amazon"]
domain = domains[0]
img_dir = "./domain_adaptation_images/%s/images/" % domain
test_dataloader = dataloader.get_dataloader(img_dir, batch_size=batch_size, train=False)

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
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(opt['seed'])
if use_cuda:
    torch.cuda.manual_seed(opt['seed'])

# 导入现有模型
classifier = main_models.Classifier()
encoder = main_models.Encoder()
discriminator=main_models.DCD(input_features=config['DCD']['input_features'],
                              h1_features=config['DCD']['h1_features'],
                              h2_features=config['DCD']['h2_features'])
attention = main_models.Attention(input_features=config['Attention']['input_features'],
                                  h_features=config['Attention']['h_features'],
                                  normalize=config['Attention']['normalize'],
                                  firstNorm=config['Attention']['first_norm'])

classifier.load_state_dict(torch.load(f"./result/wga_office_{task}/result_{experiment_number}/classifier.pth")) #,map_location=device)
encoder.load_state_dict(torch.load(f"./result/wga_office_{task}/result_{experiment_number}/encoder.pth")) #,map_location=device)
discriminator.load_state_dict(torch.load(f"./result/wga_office_{task}/result_{experiment_number}/discriminator.pth")) #,map_location=device)
attention.load_state_dict(torch.load(f"./result/wga_office_{task}/result_{experiment_number}/attention.pth")) #,map_location=device)

# # TODO: attention需要有初始化参数，感觉在0.5左右会好一点

classifier.to(device,)
encoder.to(device)
discriminator.to(device)
attention.to(device)

as_record = []

#------------------- 获取不同domain图集的attention score-------------------
for epoch in range(opt['n_epoches_3']):
    acc = 0
    deno = 0
    # discriminator.eval()
    with torch.no_grad():
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)

            # y_test_pred = classifier(encoder(data))
            encoder_test = encoder(data)
            attention_score = attention(encoder_test)

            # record
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

mkdir("result/interpretation/")
with open("result/interpretation/%s.pkl" % domain, "wb") as f:
    pickle.dump(as_record,f)

print('everything done')


# save
# name = 'wga_office_a2w_1'
# name = 'wga_office_w2a_1'
# name = 'wga_office_d2w_1'
# torch.save(encoder.state_dict(), './result/{}/encoder.pth'.format(name))
# torch.save(attention.state_dict(), './result/{}/attention.pth'.format(name))
# torch.save(classifier.state_dict(), './result/{}/classifier.pth'.format(name))
# torch.save(discriminator.state_dict(), './result/{}/discriminator.pth'.format(name))

exit()

# %%
import matplotlib.pyplot as plt
plt.pcolor(as_record)
plt.show()

