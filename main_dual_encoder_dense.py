import os
from models import dual_encoder_trainer
from test_tube import HyperOptArgumentParser

parser = HyperOptArgumentParser(strategy='grid_search')

# --------------------------
# build program arguments
# --------------------------

parser.add_opt_argument_list('--lr_1', default=0.0001, options=[0.0001, 0.0002, 0.0004, 0.0008, 0.001, 0.002], type=float, tunnable=False)

# training params
parser.add_argument('--nb_epochs', default=30, type=int)
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--optimizer_name', default='adam')
parser.add_argument('--eval_every_n_batches', default=200, type=int)
parser.add_argument('--train_mode', default='train')

# model params
parser.add_argument('--vocab_size', default=50000, type=int)
parser.add_argument('--embedding_dim', default=320, type=int)
parser.add_argument('--max_seq_len', default=50, type=int)
parser.add_argument('--nb_grams', default=2, type=int)

# dataset params
parser.add_argument('--dataset_name', default='prod_all')

# path params
parser.add_argument('--root_dir', default='/Users/waf/Desktop/angie_f')
parser.add_argument('--dataset_train_path', default='/Users/waf/Desktop/angie_f/data/train.csv')
parser.add_argument('--dataset_test_path', default='/Users/waf/Desktop/angie_f/data/test.csv')
parser.add_argument('--dataset_val_path', default='/Users/waf/Desktop/angie_f/data/val.csv')
parser.add_argument('--vocab_path', default='/Users/waf/Desktop/angie_f/data/vocabulary.txt')

parser.add_argument('--model_save_dir', default='/Users/waf/Desktop/angie_f/model_ckpt')
parser.add_argument('--test_tube_dir', default='/Users/waf/Desktop/angie_f/test_tube')

# experiment params
parser.add_argument('--exp_name', default='dual_conv_bot')
parser.add_argument('--exp_desc', default='Dual convnet + dot product loss. Base model')
parser.add_argument('--debug', default=False)

# tf params
parser.add_argument('--gpus', default='0')
# --------------------------
# --------------------------

# parse params
hparams = parser.parse_args()

# --------------------------
# TRAIN *****************
# --------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpus

dual_encoder_trainer.train_main(hparams)
