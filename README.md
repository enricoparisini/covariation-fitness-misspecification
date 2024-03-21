This repository analyses the interplay among covariation, fitness estimates and misspecification in models trained on homologous protein sequences. There are effects (mainly of phylogeny) that make the sequence distribution of a protein family inequivalent to (the exponential of) its fitness distribution, and [recent work](https://www.biorxiv.org/content/10.1101/2022.01.29.478324v2) suggests that models that are misspecified with respect to the target sequence distribution may attain better performance in fitness estimates than less misspecified models. The importance of such a counterintuitive conclusion lies in its implications for the design of future fitness estimators.

This picture calls for other approaches to test and quantify the relation between misspecification and quality of fitness estimates, and this is the main motivation driving this project. My basic assumption is that how well a model captures covariation across sites in a given protein family is a direct measure of its level of misspecification with respect to the sequence distribution. Comparing the performance in fitness of different models with their ability to reproduce marginals and correlations represents a viable approach to assess the role of misspecification in variant effect prediction and protein design. The results from the cases I have considered so far confirm and detail the conclusions of [this paper](https://www.biorxiv.org/content/10.1101/2022.01.29.478324v2).

Started as a fork of [EVE](https://github.com/OATML-Markslab/EVE), this repository contains a number of additional modules and tools to compute and manipulate marginals and correlations, as well as to sample and produce evolutionary scores from 2-site Potts models. The code to perform DCA in Mean Field Approximation closely follows the implementation in [EVcouplings](https://github.com/debbiemarkslab/EVcouplings). The notebooks `1_Sample_MSAs_and_performance_over_training`, `2_Compare_models_covariation` and `3_Evolutionary_indices_across_models` illustrate the main steps to compute and compare fitness and covariation scores. I use [this repository](https://github.com/enricoparisini/Mi3-GPU) and [this Kaggle notebook](https://www.kaggle.com/code/enricoparisini/mi3-training-and-generation) to train 2-site models and sample sequences from them.

The required environment is the same as EVE's and it may be created via conda and the provided EVE.yml file as follows:
```
  conda env create -f EVE.yml
  conda activate EVE
```
 



