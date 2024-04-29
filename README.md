<h1 align=center>
  Robust Learning of Tractable Probabilistic Models
</h1>

<p align=center>  
  Rohith Peddi, Tahrima Rahman, Vibhav Gogate
</p>

<div align=center>
  <a src="https://img.shields.io/badge/project-website-green" href="">
    <img src="https://img.shields.io/badge/project-website-green">
  </a>
  <a src="https://img.shields.io/badge/paper-arxiv-red" href="https://proceedings.mlr.press/v180/peddi22a.html">
    <img src="https://img.shields.io/badge/paper-arxiv-red">
  </a>
  <a src="https://img.shields.io/badge/bibtex-citation-blue" href="">
    <img src="https://img.shields.io/badge/bibtex-citation-blue">
  </a> 
</div>

## UPDATE

**Apr 2022:** Paper accepted at UAI 2022

## CITATION

```

@InProceedings{pmlr-v180-peddi22a,
  title = 	 {Robust learning of tractable probabilistic models},
  author =       {Peddi, Rohith and Rahman, Tahrima and Gogate, Vibhav},
  booktitle = 	 {Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {1572--1581},
  year = 	 {2022},
  editor = 	 {Cussens, James and Zhang, Kun},
  volume = 	 {180},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {01--05 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v180/peddi22a/peddi22a.pdf},
  url = 	 {https://proceedings.mlr.press/v180/peddi22a.html},
  abstract = 	 {Tractable probabilistic models (TPMs) compactly represent a joint probability distribution over a large number of random variables and admit polynomial  time computation of (1) exact likelihoods; (2) marginal probability distributions over a small subset of variables given evidence; and (3) in some cases most probable explanations over all non-observed variables given observations. In this paper, we leverage these tractability properties to solve the robust maximum likelihood parameter estimation task in TPMs under the assumption that a TPM structure and complete training data is provided as input. Specifically, we show that TPMs learned by optimizing the likelihood perform poorly when data is subject to adversarial attacks/noise/perturbations/corruption and we can address this issue by optimizing robust likelihood. To this end, we develop an efficient approach for constructing uncertainty sets that model data corruption in TPMs and derive an efficient gradient-based local search method for learning TPMs that are robust against these uncertainty sets. We empirically demonstrate the efficacy of our proposed approach on a collection of benchmark datasets.}
}
```


