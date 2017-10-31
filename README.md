# google-smart-reply-2017
Google smart reply implementation in tensorflow

## Getting started

1. Get Ubuntu corpus dataset for testing from [here](https://s3.amazonaws.com/ngv-public/data.zip)
```
wget https://s3.amazonaws.com/ngv-public/data.zip -O data.zip
```

2. Unzip and move data files wherever you want.
```
unzip data.zip -d .
```

3. Install conda environment
```bash
conda create -n sr python=3.6
pip install -r requirements.txt

source activate sr
```

4. Update the path variables with links to the data and where you want to save model output
```
# path params
parser.add_argument('--root_dir', default='')
parser.add_argument('--dataset_train_path', default='')
parser.add_argument('--dataset_test_path', default='')
parser.add_argument('--dataset_val_path', default='')
parser.add_argument('--vocab_path', default='')

parser.add_argument('--model_save_dir', default='')
parser.add_argument('--test_tube_dir', default='')
```

5. Start training
```bash
python main_dual_encoder_dense.py
```

