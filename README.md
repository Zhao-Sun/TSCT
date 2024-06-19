# TSCT

This is a Pytorch implementation of TSCT: "[Improving Long-Term Electricity Time Series Forecasting in Smart Grid with a Three-Stage Channel-Temporal Approach](xxx.pdf)". 


## Features
- [x] Support both Univariate and Multivariate long-term time series forecasting.
- [x] Support visualization of weights.
- [x] Support scripts on different look-back window size.



Beside TSCT, we provide five significant forecasting Transformers to re-implement the results in the paper.
- [x] [Transformer](https://arxiv.org/abs/1706.03762) (NeuIPS 2017)
- [x] [Informer](https://arxiv.org/abs/2012.07436) (AAAI 2021 Best paper)
- [x] [Autoformer](https://arxiv.org/abs/2106.13008) (NeuIPS 2021)
- [x] [Pyraformer](https://openreview.net/pdf?id=0EXmFzUn5I) (ICLR 2022 Oral)
- [x] [FEDformer](https://arxiv.org/abs/2201.12740) (ICML 2022)
- [x] [PatchTST](https://openreview.net/forum?id=Jbdc0vTOcol) (ICLR 2023)




## Getting Started
### Environment Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n TSCT python=3.7.13
conda activate TSCT
pip install -r requirements.txt
```



### Data Preparation

You can obtain all the nine benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well pre-processed and can be used easily.

```
mkdir dataset
```
**Please put them in the `./dataset` directory**

### Training Example
- In `scripts/ `, we provide the model implementation *Dlinear/Autoformer/Informer/Transformer*
- In `FEDformer/scripts/`, we provide the *FEDformer* implementation
- In `Pyraformer/scripts/`, we provide the *Pyraformer* implementation


## Citing

If you find this repository useful for your work, please consider citing it as follows:

```bibtex
@article{xxx,
  title={Improving Long-Term Electricity Time Series Forecasting in Smart Grid with a Three-Stage Channel-Temporal Approach},
  author={xxx},
  journal={Journal of Cleaner Production},
  year={2024}
}
```

Please remember to cite all the datasets and compared methods if you use them in your experiments.
