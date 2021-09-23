# Summary

This repository contains the python scripts used for the analysis and graphs generation that was used for the [insert pre-publication link]

Contents:

- [figures/manuscript_figures](https://github.com/otosystems/deeptone_chimps_publication/blob/master/figures/manuscript_figures): Figures used in the manuscript
- [models](https://github.com/otosystems/deeptone_chimps_publication/blob/master/models):
    - [manuscript_experiments.py](https://github.com/otosystems/deeptone_chimps_publication/blob/master/models/manuscript_experiments.py) Calls the funcionality in [deeptone_classifier.py](https://github.com/otosystems/deeptone_chimps_publication/blob/master/models/deeptone_classifier.py) to taylor the experiment discussed in the manuscript.
    - [deeptone_classifier.py](https://github.com/otosystems/deeptone_chimps_publication/blob/master/models/deeptone_classifier.py) Contains the main functionality used for the analysis
- [results_folder](https://docs.conda.io/en/latest/): Contains the .csv files containg the main results used in the manuscript
- requirements.txt: List of dependcies required to run the experiment.

The functionality for:

    a) Importing audio 
    b) Generating embeddings spaces
    c) Generating machine / deep learning classifications
    d) Plotting
    
are all contained in [models.deeptone_classifier.py](https://github.com/otosystems/deeptone_chimps_publication/blob/master/models/deeptone_classifier.py).


Instead the functionality in [models.manuscript_experiments.py](https://github.com/otosystems/deeptone_chimps_publication/blob/master/models/manuscript_experiments.py) is dedicated to choosing hyperparameters relative to the functionality in [models.deeptone_classifier.py](https://github.com/otosystems/deeptone_chimps_publication/blob/master/models/deeptone_classifier.py), and combining the enclosed functionality to create the specific experiments and respective graphs discussed in the manuscript. 

To run the experiment discussed in the manuscript please install the required dependcies listed in requirements.txt.

We reccomend doing this in a [conda](https://docs.conda.io/en/latest/) virtual environmet

```bash
conda create --name <env> --file requirements.txt 
conda activate <env>
```

The experiment and subsequent graphs used in the manuscript can then be generated by running

```bash
python models.manuscript_experiments.py
```





