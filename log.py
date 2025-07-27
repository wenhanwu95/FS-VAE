import argparse
import os

parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--name', type=str, default='', help="log name")
parser.add_argument('--ntu', type=int, help="num_classes")
parser.add_argument('--ss', type=int, help="split size")
parser.add_argument('--tm', type=str, help='text mode')
parser.add_argument('--za', type=str, default='' , help='zsl acc')
parser.add_argument('--ua', type=str, help='unseen acc')
parser.add_argument('--sa', type=str, help='seen acc')
parser.add_argument('--hm', type=str, help='h mean')
parser.add_argument('--le', type=str, help='language encoder arch')
parser.add_argument('--ls', type=str, help='latent embedding shape')
parser.add_argument('--nepc', type=str, help='epochs')
args = parser.parse_args()

le = args.le.split('/')[-1]
name = f'ntu{args.ntu}-B{le}-ls{args.ls}-nepc{args.nepc}'

if not os.path.exists('log'):
    os.makedirs('log')

with open(f'log/{name}.txt', 'a') as f:
    # f.write(f'ntu{args.ntu}, B{le}, ls{args.ls}, nepc{args.nepc}')
    f.write(f'zsl_acc: {args.za}, seen_acc: {args.sa}, unseen_acc: {args.ua}, h_mean: {args.hm},    in {args.ntu}u{args.ss}-{args.tm}\n')
