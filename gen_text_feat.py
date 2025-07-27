import argparse
import os

import numpy as np
import torch

from text_extractor.text_encoder import Text_Encoder
from text_extractor.tokenizer import Tokenizer

parser = argparse.ArgumentParser(description='View adaptive')
# parser.add_argument('--ntu', type=int, help='ntu')
parser.add_argument('--ntu', type=int, default=120, help='ntu')
parser.add_argument('--arch', type=str, default='ViT-B/32', help='text extractor architecture')
parser.add_argument('--save_dir', type=str, default='./text_feats', help='ntu')
parser.add_argument('--gpu', type=int, default=0, help='gpu num')
args = parser.parse_args()


nl = ['lb', 'ad', 'md']
nc = args.ntu
save_dir = args.save_dir + f'/{args.arch}'


class Text_Generator:
    def __init__(self, args, save_path):
        self.device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.set_device(self.device)
        self.save_path = save_path
        
        T = Tokenizer(nc, nl)
        self.token_dict = T.gen_token()
        self.text_encoder = Text_Encoder(args.arch)
        self.text_encoder.to(self.device)
        
    def gen_text_feature(self):
        print('Generating Text Feature ...')
        self.text_encoder.eval()
        with torch.no_grad():
            for k in self.token_dict.keys():
                print(f'Processing {k}_{nc} ...')
                t_inp = self.token_dict[k].to(self.device)
                tf = self.text_encoder(t_inp)
                np.save(self.save_path[k], tf.cpu().numpy())
        
        torch.cuda.empty_cache()
        print('Finish generating!')

if __name__ == '__main__':
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = {}
    
    for k in nl:
        if not os.path.isfile(f'{save_dir}/{k}_{args.ntu}.npy'):
            save_path[k] = f'{save_dir}/{k}_{args.ntu}'
    
    if save_path != {}:
        tg = Text_Generator(args, save_path)
        tg.gen_text_feature()
