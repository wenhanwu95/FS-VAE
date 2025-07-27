# Thanks to P Gupta for the released code on github (https://github.com/skelemoa/synse-zsl)
import argparse
import os
import pdb
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm

from data_cnn60 import (AverageMeter, NTUDataLoaders, get_cases,
                        get_num_classes, make_dir)
from model import (MLP, Decoder, Encoder, KL_divergence, Wasserstein_distance,
                   reparameterize)
#print("after import", flush=True)
#exit()
# Arg Parser
parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--ss', type=int, help="split size")
parser.add_argument('--st', type=str, help="split type")
parser.add_argument('--dataset', type=str, help="dataset path", default="/lstr/sahara/flow/zg/nturgb+d_skeletons")
parser.add_argument('--wdir', type=str, help="directory to save weights path")
parser.add_argument('--le', type=str, help="language embedding model")
parser.add_argument('--ve', type=str, help="visual embedding model")
parser.add_argument('--phase', type=str, help="train or val")
parser.add_argument('--gpu', type=str, help="gpu device number")
parser.add_argument('--ntu', type=int, help="ntu120 or ntu60")
parser.add_argument('--num_cycles', type=int, help="no of cycles")
parser.add_argument('--num_epoch_per_cycle', type=int, help="number_of_epochs_per_cycle")
parser.add_argument('--latent_size', type=int, help="Latent dimension")
parser.add_argument('--mode', type=str, help="Mode")
parser.add_argument('--load_epoch', type=int, help="load epoch", default=None)
parser.add_argument('--load_classifier', action='store_true')
parser.add_argument('--tm', type=str, help='text mode')
parser.add_argument('--alpha', type=float, help='weight for calibrated loss')     #added arg
parser.add_argument('--lmd', type=float, help='temperature in calibrated loss', default=1.0) #added arg
parser.add_argument('--use_cr_fact', type=int, help='use_cr_fact', default=0) #added arg
parser.add_argument('--version', type=str, help='v', default="NA") #added arg
args = parser.parse_args()

gpu = args.gpu
ss = args.ss
st = args.st
dataset_path = args.dataset
wdir = args.wdir
le = args.le
ve = args.ve
phase = args.phase
num_class = args.ntu
num_cycles = args.num_cycles
cycle_length = args.num_epoch_per_cycle
latent_size = args.latent_size
load_epoch = args.load_epoch
mode = args.mode
load_classifier = args.load_classifier
tm = args.tm

# Embedding Dim
vis_emb_input_size = 256
text_emb_input_size = 1024

os.environ["CUDA_VISIBLE_DEVICES"] = gpu
seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda")

if not os.path.exists(f'{wdir}/{le}/{tm}'):
    os.makedirs(f'{wdir}/{le}/{tm}')

# DataLoader
ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
train_loader = ntu_loaders.get_train_loader(64, 0) # bs, num_workers
zsl_loader = ntu_loaders.get_val_loader(64, 0)
val_loader = ntu_loaders.get_test_loader(64, 0)
train_size = ntu_loaders.get_train_size()
zsl_size = ntu_loaders.get_val_size()
val_size = ntu_loaders.get_test_size()
print('Train on %d samples, validate on %d samples' % (train_size, val_size))

if phase == 'val':
    gzsl_inds = np.load(f'label_splits/{st}s{str(num_class - ss)}.npy')
    unseen_inds = np.sort(np.load(f'label_splits/{st}v{str(ss)}_0.npy'))
    seen_inds = np.load(f'label_splits/{st}s{str(num_class - ss - ss)}_0.npy')
else:
    gzsl_inds = np.arange(num_class)
    unseen_inds = np.sort(np.load(f'label_splits/{st}u{str(ss)}.npy'))
    seen_inds = np.load(f'label_splits/{st}s{str(num_class - ss)}.npy')

tml = tm.split('_')
tfl = [torch.from_numpy(np.load(f'./text_feats/{le}/{m}_{num_class}.npy')) for m in tml]
text_feat = torch.concat(tfl, dim=-1)
text_emb_input_size = text_feat.size(-1)
text_emb = text_feat / torch.norm(text_feat, dim=1, keepdim=True).repeat([1, text_emb_input_size])

unseen_text_emb = text_emb[unseen_inds, :]
seen_text_emb = text_emb[seen_inds, :]
print("language embeddings loaded.")

# VAE
sequence_encoder = Encoder([vis_emb_input_size, latent_size]).to(device)
sequence_decoder = Decoder([latent_size, vis_emb_input_size]).to(device)
text_encoder = Encoder([text_emb_input_size, latent_size]).to(device)
text_decoder = Decoder([latent_size, text_emb_input_size]).to(device)

# Optimizer
params = []
for model in [sequence_encoder, sequence_decoder, text_encoder, text_decoder]:
    params += list(model.parameters())
optimizer = optim.Adam(params, lr = 0.0001)

# Loss
criterion1 = nn.MSELoss().to(device)

def get_text_data(target):
    return text_emb[target.cpu()].view(target.shape[0], text_emb_input_size).float()

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_models(load_epoch):
    se_checkpoint = f'{wdir}/{le}/{tm}/se_{str(load_epoch)}.pth.tar'
    sd_checkpoint = f'{wdir}/{le}/{tm}/sd_{str(load_epoch)}.pth.tar'
    te_checkpoint = f'{wdir}/{le}/{tm}/te_{str(load_epoch)}.pth.tar'
    td_checkpoint = f'{wdir}/{le}/{tm}/td_{str(load_epoch)}.pth.tar'

    sequence_encoder.load_state_dict(torch.load(se_checkpoint)['state_dict'])
    sequence_decoder.load_state_dict(torch.load(sd_checkpoint)['state_dict'])
    text_encoder.load_state_dict(torch.load(te_checkpoint)['state_dict'])
    text_decoder.load_state_dict(torch.load(td_checkpoint)['state_dict'])

def train_one_cycle(cycle_num, cycle_length): # 0-10, 1700
    # root_path = "/lstr/sahara/flow/zg/"
    # output_filename = root_path + "msf_cali/output/" + "version_" + args.version \
    #                   + "_alpha_" + str(args.alpha) \
    #                   + "_use_cr_" + str(args.use_cr_fact) + "_lmd_" \
    #                   + str(args.lmd)  
    s_epoch = (cycle_num)*(cycle_length) # 0, 1700, 3400,...
    e_epoch = (cycle_num+1)*(cycle_length) # 1700, 3400, 5100, ... 
    if cycle_length == 1700:
        cr_fact_epoch = 1400
    else:
        cr_fact_epoch = 1500
        
    for epoch in range(s_epoch, e_epoch):
        losses = AverageMeter()
        ce_loss_vals = []

        # verb models
        sequence_encoder.train()
        sequence_decoder.train()
        text_encoder.train()
        text_decoder.train()

        # hyper params
        k_fact = max((0.1*(epoch- (s_epoch+1000))/3000), 0)
        cr_fact = 1*(epoch > (s_epoch + cr_fact_epoch))
        k_fact2 = max((0.1*(epoch - (s_epoch + cr_fact_epoch))/3000), 0)*(cycle_num>1)
        
        (inputs, target) = next(iter(train_loader))
        s = inputs.to(device)
        t = target.to(device)
        t = get_text_data(t).to(device)

        smu, slv = sequence_encoder(s)
        sz = reparameterize(smu, slv)
        sout = sequence_decoder(sz)

        tmu, tlv = text_encoder(t)
        tz = reparameterize(tmu, tlv)
        tout = text_decoder(tz)

        sfromt = sequence_decoder(tz)
        tfroms = text_decoder(sz)

        s_recons = criterion1(s, sout)
        t_recons = criterion1(t, tout)
        s_kld = KL_divergence(smu, slv).to(device) 
        t_kld = KL_divergence(tmu, tlv).to(device)

        ################################################Calibrated Loss################################################
        loss = 0
        
        t_perm = torch.randperm(t.shape[0]) 
        t_cali = 0
        t_cali_temp0 = torch.sum(torch.square(t - tfroms[t_perm]), 1) \
                       - torch.sum(torch.square(t - tfroms), 1)         
        t_cali_temp1 = 1.0/(1.0 + torch.exp((t_cali_temp0)/args.lmd))
        t_cali += torch.mean(t_cali_temp1) 
        '''         
        for i in range(t.shape[0]): # or just (range(batch_size))
            pos_pair = t[i] - tfroms[i] # expected dimension: feature dimension
            if epoch == 0 and i==0:
                with open(output_filename+".txt", "a") as file:
                    file.write("Epoch " + str(epoch) + \
                               "; pos_pair.shape: " + str(pos_pair.shape) + '\n') 
            # neg_pairs = t[i] - tfroms # expected dimension: batch_size*feature dimension
            #t_cali_temp0 = torch.sum(torch.square(neg_pairs)) - torch.sum(torch.square(pos_pair))
            t_cali_temp0 = - torch.sum(torch.square(pos_pair))
            # t_cali_temp0: expected dimension: batch_size*feature dimension
            t_cali_temp1 = 1.0/(1.0 + torch.exp(t_cali_temp0/args.lmd)) 
            # t_cali_temp1 expected dimension: batch_size 
            t_cali += torch.mean(t_cali_temp1)
        ''' 
        """
        t_cali = 0
        for i in range(t.shape[0]): # or just (range(batch_size))
            pos_pair = t[i] - t_froms[i] # expected dimension: feature dimension
            for j in range (t.shape[0]):
                neg_pair = t[i] - t_froms[j] # expected dimension: feature dimension
                t_cali += 1/t.shape[0] * 1.0/(1.0 + torch.exp(torch.sum(torch.square(neg_pair)) - torch.sum(torch.square(pos_pair))))
        """
        s_cali = 0
        s_perm = torch.randperm(s.shape[0]) 
        '''
        for i in range(s.shape[0]): # or just (range(batch_size))
            pos_pair = s[i] - sfromt[i] # expected dimension: feature dimension
            #neg_pairs = s[i] - sfromt # expected dimension: batch_size*feature dimension
            #s_cali_temp = 1.0/(1.0 + torch.exp(torch.sum(torch.square(neg_pairs)) - torch.sum(torch.square(pos_pair)))) 
            s_cali_temp = 1.0/(1.0 + torch.exp(- torch.sum(torch.square(pos_pair))/args.lmd)) 
            # t_cali_temp expected dimension: batch_size 
            s_cali += torch.mean(s_cali_temp)
        '''  
        s_cali_temp0 = - torch.sum(torch.square(s - sfromt), 1)          
        s_cali_temp1 = 1.0/(1.0 + torch.exp((s_cali_temp0)/args.lmd))
        s_cali += torch.mean(s_cali_temp1) 

        if args.use_cr_fact == 1:
            coef = args.lmd*args.alpha*cr_fact # consider adding a hyper-parameter coef to tune
        else:
            coef = args.lmd*args.alpha
        loss += coef*(t_cali + s_cali) 
          

        t_crecons = criterion1(t, tfroms)
        s_crecons = criterion1(s, sfromt)

        #cr_fact = 1.0
        loss += s_recons + t_recons 
        loss -= k_fact*(s_kld) + k_fact2*(t_kld)
        #loss += cr_fact*(s_crecons) + cr_fact*(t_crecons)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        ce_loss_vals.append(loss.cpu().detach().numpy())
        # if epoch % 500 == 0:
        #     print('---------------------')
        #     print('Epoch-{:<3d} \t loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, loss=losses))
        # #     print('srecons {:.4f}\t ntrecons {:.4f}\t'.format(s_recons.item(), t_recons.item()))
        # #     print('skld {:.4f}\t tkld {:.4f}\t'.format(s_kld.item(), t_kld.item()))
        # #     print('screcons {:.4f}\t tcrecons {:.4f}\t'.format(s_crecons.item(), t_crecons.item()))        

    return

def save_model(epoch):
    se_checkpoint = f'{wdir}/{le}/{tm}/se_{str(epoch)}.pth.tar'
    sd_checkpoint = f'{wdir}/{le}/{tm}/sd_{str(epoch)}.pth.tar'
    te_checkpoint = f'{wdir}/{le}/{tm}/te_{str(epoch)}.pth.tar'
    td_checkpoint = f'{wdir}/{le}/{tm}/td_{str(epoch)}.pth.tar'
    
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': sequence_encoder.state_dict(),
        'optimizer': optimizer.state_dict()
    }, se_checkpoint)
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': sequence_decoder.state_dict(),
    }, sd_checkpoint)
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': text_encoder.state_dict(),
    }, te_checkpoint)
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': text_decoder.state_dict(),
    }, td_checkpoint)

def train_classifier():
    cls = MLP([latent_size, ss]).to(device)
    if load_classifier == True:
        cls_checkpoint = f'{wdir}/{le}/{tm}/clasifier.pth.tar'
        cls.load_state_dict(torch.load(cls_checkpoint)['state_dict'])
    else:
        cls_optimizer = optim.Adam(cls.parameters(), lr = 0.001)
        # print('---------------------')
        # print('classifier_training ....')
        with torch.no_grad():
            n_t = unseen_text_emb.to(device).float()
            n_t = n_t.repeat([500, 1])
            y = torch.tensor(range(ss)).to(device)
            y = y.repeat([500])
            text_encoder.eval()
            t_tmu, t_tlv = text_encoder(n_t)
            t_z = reparameterize(t_tmu, t_tlv)

        criterion2 = nn.CrossEntropyLoss().to(device)
        best = 0
        
        for c_e in range(300):
            cls.train()
            out = cls(t_z)
            c_loss = criterion2(out, y)
            cls_optimizer.zero_grad()
            c_loss.backward()
            cls_optimizer.step()
            c_acc = float(torch.sum(y == torch.argmax(out, -1)))/(ss*500)
            # if c_e % 100 == 0:
            #     print("Train Loss :", c_loss.item(), "Train Accuracy:", c_acc)

    cls.eval()

    u_inds = torch.from_numpy(unseen_inds)
    final_embs = []
    with torch.no_grad():
        sequence_encoder.eval()
        cls.eval()
        count = 0
        num = 0
        preds = []
        tars = []
        for (inp, target) in zsl_loader:
            t_s = inp.to(device)
            nt_smu, t_slv = sequence_encoder(t_s)
            final_embs.append(nt_smu)
            t_out = cls(nt_smu)
            pred = torch.argmax(t_out, -1)
            preds.append(u_inds[pred.cpu()])
            tars.append(target)
            count += torch.sum(u_inds[pred.cpu()] == target)
            num += len(target)

    zsl_accuracy = float(count)/num
    final_embs = np.array([j.cpu().numpy() for i in final_embs for j in i])
    p = [j.item() for i in preds for j in i]
    t = [j.item() for i in tars for j in i]
    p = np.array(p)
    t = np.array(t)
    
    val_out_embs = []
    with torch.no_grad():
        sequence_encoder.eval()
        cls.eval()
        gzsl_count = 0
        gzsl_num = 0
        gzsl_preds = []
        gzsl_tars = []
        loader = val_loader if phase == 'train' else zsl_loader
        for (inp, target) in loader:
            t_s = inp.to(device)
            t_smu, t_slv = sequence_encoder(t_s)
            t_out = cls(t_smu)
            val_out_embs.append(F.softmax(t_out, 1))
            pred = torch.argmax(t_out, -1)
            gzsl_preds.append(u_inds[pred.cpu()])
            gzsl_tars.append(target)
            gzsl_count += torch.sum(u_inds[pred.cpu()] == target)
            num += len(target)
    
    val_out_embs = np.array([j.cpu().numpy() for i in val_out_embs for j in i])
    
    return zsl_accuracy, val_out_embs, cls

def get_seen_zs_embeddings(cls):
    final_embs = []
    out_val_embeddings = []
    u_inds = torch.from_numpy(unseen_inds)
    with torch.no_grad():
        sequence_encoder.eval()
        cls.eval()
        count = 0
        num = 0
        preds = []
        tars = []
        for (inp, target) in val_loader:
            t_s = inp.to(device)
            t_smu, t_slv = sequence_encoder(t_s)
            final_embs.append(t_smu)
            t_out = cls(t_smu)
            out_val_embeddings.append(F.softmax(t_out))
            pred = torch.argmax(t_out, -1)
            preds.append(u_inds[pred.cpu()])
            tars.append(target)
            count += torch.sum(u_inds[pred.cpu()] == target)
            num += len(target)

    out_val_embeddings = np.array([j.cpu().numpy() for i in out_val_embeddings for j in i])
    return out_val_embeddings

def save_classifier(cls):
    cls_checkpoint = f'{wdir}/{le}/{tm}/classifier.pth.tar'
    save_checkpoint({'state_dict': cls.state_dict()}, cls_checkpoint)


if __name__ == "__main__":
    best = 0
    if mode == 'eval':
        if load_epoch != None:
            load_models(load_epoch)
            zsl_acc, val_out_embs, _ = train_classifier()
            print('zsl accuracy ', zsl_acc)
        else:
            print('Mention Epoch to Load')
    else:
        if load_epoch != None:
            load_models(load_epoch)
        else:
            load_epoch = -1
        for num_cycle in range((load_epoch+1)//cycle_length, num_cycles):
            train_one_cycle(num_cycle, cycle_length)
            if phase == 'train':
                save_model(cycle_length*(num_cycle+1)-1)
            zsl_acc, val_out_embs, cls = train_classifier()
            
            if (zsl_acc > best):
                best = zsl_acc
                save_classifier(cls)
                print('---------------------')
                print(f'zsl_accuracy increased to {best :.2%} on cycle ', num_cycle)
                print('checkpoint saved')
                if phase == 'train':
                    np.save(f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_gzsl_zs.npy', val_out_embs)
                else:
                    np.save(f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_unseen_zs.npy', val_out_embs)
                    seen_zs_embeddings = get_seen_zs_embeddings(cls)
                    np.save(f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_seen_zs.npy', seen_zs_embeddings)
