# CSMGAN
Code for ACM MM2020 paper

**Jointly Cross- and Self-Modal Graph Attention Network for Query-Based Moment Localization** <br />
**[[Paper](https://arxiv.org/abs/2008.01403)]** <br />

### Main Results
##### Activity Caption
| R@1, IoU=0.3      | R@1, IoU=0.5     | R@1, IoU=0.7      | R@5, IoU=0.3      | R@5, IoU=0.5     | R@5, IoU=0.7      |
| ---------- | :-----------:  | :-----------: | :-----------:  | :-----------: | :-----------:  |
| 68.52     | 49.11    | 29.15     | 87.68     | 77.43    | 59.63     |
##### TACoS
| R@1, IoU=0.1      | R@1, IoU=0.3     | R@1, IoU=0.5      | R@5, IoU=0.1      | R@5, IoU=0.3     | R@5, IoU=0.5      |
| ---------- | :-----------:  | :-----------: | :-----------:  | :-----------: | :-----------:  |
| 42.74     | 33.90    | 27.09     | 68.97     | 53.98    | 41.22     |
##### Charades-STA
| R@1, IoU=0.5      | R@1, IoU=0.7     |R@5, IoU=0.5      | R@5, IoU=0.7     |
| ---------- | :-----------:  | :-----------: | :-----------:  |
| 60.04     | 37.34    | 89.01     | 61.85     |
##### DiDeMo
| R@1, IoU=0.5      | R@1, IoU=0.7     |R@5, IoU=0.5      | R@5, IoU=0.7     |
| ---------- | :-----------:  | :-----------: | :-----------:  |
| 29.44     | 19.16    | 70.77     | 41.61     |

### Prerequisites
* Python 3.6
* Pytorch >= 0.4.0

### Preparation
* Download Pretrained [Glove Embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip)
* Download Extracted Features of [Activity Caption](http://activity-net.org/challenges/2016/download.html)
* Download Extracted Features of [TACoS](https://drive.google.com/file/d/1kK_FTo6USmPhO1vam3uvBMtJ3QChUblm/view)
* Download Pretrained model on [Google Drive](https://drive.google.com/drive/folders/149wIt533qSnrY_rgDaqfufL-Kn58aLw5?usp=sharing)

### Evaluation
    $ python main.py --word2vec-path /yourpath/glove_model.bin --dataset ActivityNet --feature-path /yourpath/ActivityCaptions/ActivityC3D --train-data data/activity/train_data_gcn.json --val-data data/activity/val_data_gcn.json --test-data data/activity/test_data_gcn.json --max-num-epochs 20 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-06 --lr 8e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-7 --evaluate --model-load-path ./models_activity/model_6852
    $ python main.py --word2vec-path /yourpath/glove_model.bin --dataset TACOS --feature-path /yourpath/TACOS/TACOS --train-data data/tacos/TACOS_train_gcn.json --val-data data/tacos/TACOS_val_gcn.json --test-data data/tacos/TACOS_test_gcn.json --max-num-epochs 40 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-07 --lr 4e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-8 --evaluate --model-saved-path models_tacos --batch-size 64 --model-load-path ./models_tacos/model_4274

### Citation
If you use this code please cite:

```
@inproceedings{liu2020jointly,
    title={Jointly Cross- and Self-Modal Graph Attention Network for Query-Based Moment Localization},
    author={Liu, Daizong and Qu, Xiaoye and Liu, Xiaoyang and Dong, Jianfeng and Zhou, Pan and Xu, Zichuan},
    booktitle={Proceedings of the 28th ACM International Conference on Multimedia (MM’20)},
    year={2020}
}
```
