This code repository analyses the interplay among covariation, fitness estimates and misspecification in models trained on sequences belonging to the same protein family. There are effects (mainly of phylogenetic nature) that make the data distribution of a sequence family we observe today inequivalent to its fitness distribution, and [recent work](https://www.biorxiv.org/content/10.1101/2022.01.29.478324v2) suggested that models that misspecify the target density distribution they are trained on may attain better performance in fitness estimates than sharply-fitted models. The importance of such a counterintuitive conclusion lies in its obvious implications for the design of future fitness estimators.

This picture calls for other approaches to test and quantify the relation between misspecification and quality of fitness estimates, and this is the main motivation driving this project. My starting point is that how well a model captures the covariation among sites of a protein family is a direct measure of how much it is misspecifying the density distribution of that family. Comparing the performance of different models in reproducing marginals and correlations with their performance in fitness is thus another effective way to assess the role of misspecification in variant effect prediction and protein design. The examples I have considered so far seem to confirm that sharply-fitted models that misspecify less are typically worse at estimating fitness, in line with the conclusions in [this paper](https://www.biorxiv.org/content/10.1101/2022.01.29.478324v2).

Started as a fork of [EVE](https://github.com/OATML-Markslab/EVE), I have added a number of modules and tools to compute and manipulate marginals and correlations, as well as to sample and produce evolutionary scores from 2-site models. The notebooks `1_Sample_MSAs_and_performance_over_training`, `2_Compare_models_covariation` and `3_Evolutionary_indices_across_models` illustrate the main features and steps of the pipeline.

Just like for EVE, the required environment may be created via conda and the provided protein_env.yml file as follows:
```
  conda env create -f protein_env.yml
  conda activate protein_env
```
 



