# CoNT
Fairseq Code for Neurips 2022 paper:  "CoNT: Contrastive Neural Text Generation"


This is the [fairseq-based](https://github.com/facebookresearch/fairseq) implementation 
for NeurIPS 2022  paper: *[CoNT: Contrastive Neural Text Generation](https://arxiv.org/abs/2205.14690)*.
CoNT is a strong contrastive learning framework for neural text generation which outperforms the MLE based training method on **five** generation tasks, including *machine translation*, *summarization*, *code comment generation*, *data-to-text generation*, *commensense generation*. 
This repo is mainly about machine translation tasks. For other tasks please refer to our [transformers repo](https://github.com/Shark-NLP/CoNT)

-----

## Dependencies
Main libraries
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.7 +
- [fairseq](https://github.com/facebookresearch/fairseq) 1.0.0

```
# clone our repo and fairseq
git clone https://github.com/ChenxinAn-fdu/CoNT.git
git clone https://github.com/facebookresearch/fairseq.git
# replace the fairseq folder with our custom code
rm -rf fairseq/fairseq && mv CoNT/fairseq  fairseq/
mv fairseq CoNT/ && cd CoNT/fairseq && pip install -e . && cd ..
```

Please follow the [instruction](https://github.com/facebookresearch/fairseq/tree/main/examples/translation#wmt14-english-to-german-convolutional) in Fairseq to prepare the data.

We have provided the training scripts for IWSLT14 and WMT14 translation tasks which can make it very easy to reproduce our results: `run_iwslt14.py` and `run_wmt14.py`.

## Generating binarized_dataset
```
python run_iwslt14.py --mode preprocess
```

## Training

For example, **if you have 4 V100-32 GPUs**, run the following script for training with warmup:
```bash
python run_iwslt14.py --mode train --gpus 0,1,2,3 --warmup
```
If the `--save-dir` has already has a warmed-up checkpoint, you can directly omit the `--warmup` option 
```bash
python run_iwslt14.py --mode train --gpus 0,1,2,3
```

## Testing
```
python run_iwslt14.py --mode gen --gpus 0 --save_path /path/to/checkpoints/checkpoint_best.pt
```
With checkpoints average
```
python run_iwslt14.py --mode gen --gpus 0 --save_path /path/to/checkpoints/ --avg_ckpt
```
## Compound splitting
Previous work usually reports the results with compound splitting on WMT'14 En-De translation task. To apply compound splitting to the reference file and output file, please run the following cmd:
```
python run_wmt14.py --mode score --out_file /path/to/checkpoints/checkpoint_best.out
```

### Citing
```
@article{an2022cont,
  title={CoNT: Contrastive Neural Text Generation},
  author={An, Chenxin and Feng, Jiangtao and Lv, Kai and Kong, Lingpeng and Qiu, Xipeng and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2205.14690},
  year={2022}
}
```
