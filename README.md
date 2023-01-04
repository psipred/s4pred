![S4PRED Logo](./images/s4pred.png)

<p align="center">
	<a href="CHANGELOG.md">
        <img src="https://img.shields.io/badge/version-1.1.0-green?style=flat-square&logo=appveyor.svg" alt="Version">
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-GPL--3.0-blue?style=flat-square&logo=appveyor.svg" alt="GPL-3.0 License">
    </a>
</p>

<br>


 
# S4PRED

A tool for accurate prediction of a protein's secondary structure from only it's amino acid sequence. 

This repo contains the fully trained inference model built in Pytorch. Its a simple one line command to run it on a FASTA file containing a protein sequence. S4PRED is a product of our paper [Increasing the accuracy of single sequence prediction methods using a deep semi-supervised learning framework](https://academic.oup.com/bioinformatics/article/37/21/3744/6313164) published in Bioinformatics.

We also provide the 1.08M example pseudo-labelled training set used for training S4PRED. This has been cross-validated against the CB513 test set using a variety of sequence-searching-based methods, as real structure labels aren't available. Proper cross-validation when working with large sets of protein sequences is incredibly important, especially when working with powerful parametric models like deep neural networks.

S4PRED is a state-of-the-art single-sequence model meaning it doesn't use homology information to make predictions, only the primary amino acid sequence. 



## Requirements

- Python 3.7 or greater
- Pytorch 1.5.1 
- Biopython 1.78 or greater

The script hasn't been tested on newer Pytorch builds but it should run without any issues. Likewise, most reasonably contemporary Biopython versions should be sufficient, barring some of the oldest.  

## Installation

Clone the directory
```bash
git clone https://github.com/psipred/s4pred
cd s4pred
```
Now we download a tarball of the model weights from our public server and then extract.
```bash
wget http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz
tar -xvzf weights.tar.gz
``` 
This leaves you with a `weights/` directory containing the five models. Each weight file is ~86MB and so all together make up roughly 430MB uncompressed (~395M compressed). If you would like to check it, the MD5 of the tarball is `e04ad7d10b61551f7e07a86b65bb88dc`. If you have python and Pytorch installed you should be ready to go. 

## Usage

The S4PRED model can be used to predict a sequence's secondary structure with the following:
```bash
python run_model.py YOUR_FASTA.fas > YOUR_OUTPUT.ss2
```
The results of the prediction are piped to `stdout` and prediction should take less than a second. 
Another example:
```bash
python run_model.py --device gpu --save-files --outdir /home/the_user/s4pred_preds/ --save-by-idx ./example/multi_seqs.ss2
```
This produces predictions, using an available GPU, for the three sequences in the example FASTA file provided in `./example/multi_seqs.fas`. These are saved as `s4_out_0.ss2`, `s4_out_1.ss2`, and `s4_out_2.ss2` in  the directory `/home/the_user/s4pred_preds/`.


### Input Sequence File

The expected input is a FASTA formatted file with one or more sequences in it, `YOUR_FASTA.fas` in the example above. S4PRED will produce seperate predictions for each sequence contained in the FASTA file. See the different options below for how predictions are saved en masse. This is different from the first version of S4PRED (v1.0.0) which operated on files containing a single sequence. 

### Optional Inputs

There are several optional arguments you can give. Running `python run_model.py -h` from a terminal will print out all options with short descriptions. More explanation is included below:

- `--device`
    - This can be either `cpu` or `gpu`. 
This specifies if you want to run the model on the GPU or the CPU. By default it uses the CPU and in the vast majority of cases this will be sufficient. Also, the model should use less than 1GB of system memory. 

- `--outfmt`
    - This can be either `ss2` or `fas`. 
This specifies which output format to use. `ss2` is the default and it corresponds to the PSIPRED vertical format (PSIPRED VFORMAT). Here is an example of what it looks like:
```
# PSIPRED VFORMAT (S4PRED V1.0)

   1 M C   0.997  0.000  0.002
   2 G C   0.984  0.000  0.016
   3 D C   0.812  0.001  0.187
   ...
```
A full example output of this file is located in `examples/1qys.ss2`.

The alternative `fas` output returns the sequence FASTA file with the predicted secondary structure concatenated on a second line. This is similar to the FASTA flat-file that the RCSB PDB provides for all sequences and their DSSP based secondary structure (Downloaded from [here](https://cdn.rcsb.org/etl/kabschSander/ss.txt.gz)). We note that it doesn't appear that the PDB is continuing to provide up-to-date versions of the flat file. Here is an example:
```
>1QYS_1|Chain A|TOP7|Computationally Designed Sequence
MGDIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQL
CCCEEEEEEECCCCCEEEEEEEECCHHHHHHHHHHHHHHHHHCCCCEEEEEEEECCHHHHHHHHHHHHHHHHHCCCCEEEEEEECCEEEEEEEC
```
The above example output of this file is located in `examples/1qys_ss.fas`.

- `--fas-conf`
    - Including this flag has S4PRED output the 3-class confidence scores (i.e. those output in the `.ss2` format) as three additional lines if using `.fas` output. As the second line is the sequence, and the third line is the class assignment, the fourth through sixth lines are the loop, helix, and strand probabilities respectively. 

- `--silent`
    - Flag to suppress printing predictions to stdout.
    
- `--save-files` 
    - Flag to save each input sequence prediction in an individual file. Makes and saves to a directory called `preds/` in the same dir as this script unless --outdir is specified. **Note:** without the  `--save-by-idx` described below, the files are saved using the name record that Biopython extracts from the header line of the FASTA file. It is common for this to produce somewhat messy file names (*TO DO:* add name and ID post processing). 

- `--outdir p`
    - Absolute file-path `p`, where files are to be saved. If --save-files is used. If not specified, it defaults to making a new directory in the S4PRED directory called `preds/` and then saves sequence predictions in that dir. 
    
- `--save-by-idx`
    - If saving with --save-files, use a counter to name files instead of sequence ID. This uses the default file name prefix of `s4_out_` meaning the files are saved as `s4_out_0.ss2` or `s4_out_0.fas` for the first sequence in a FASTA file. 

### Example Run
The following is an example run on the sequence of TOP7 (PDB ID: 1QYS) using the GPU and output to the FASTA like format. The corresponding FASTA input file is located in `examples/1qys.fas` (this is the PDB FASTA file stripped of the 6-HIS tag on the C-Terminus). 
```bash
python run_model.py --device gpu --outfmt fas ./example/1qys.fas > 1qys_ss.fas
```


## Dataset
We have made the pseudo-labelled training set available to download from our public server. 
These are in a simple FASTA flat file, `s4pred_train.fas`.
```bash
wget http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/s4pred_train.fas
``` 

There are 1080886 examples in the set and the contents of the flat file look like this:
```
>A0A1J5K7I4
MYVCVCNGITEEMLDTAQKQGLSDREILNRLGVGNSCGVCVIDALDNMRSNSLKSQKTSNRKDSKKS
CEEEECCCCCHHHHHHHHHCCCCHHHHHHHHCCCCCCCHHHHHHHHHHHHHHHHHHHHHCCCCCCCC
>A0A2S4ZHT0
MRPDLMGPGFVQVLARTTLVSRHRPRQQLPSGTVRHAMVQVTERFFSHGPHRPAPGFTPMVRQRYCCA
CCCCCCCCCHHHHHEECEECCCCCCHHCCCCHHHHHEEEEEEHHHHHCCCCCCCCCCCHHHCCCCCCC
>A0A076ETI5
MGALSPSHWAIIAVVLVVLFGSKKLPDAARGLGRSMRILKTEVGELQADAPELEK
CCCCCHHHHHHHHHHHHHHHCCCCCHHHHHHHHHHHHHHHHHHHHHHCCCHHHCC
...
```
The examples go label, then sequence on a new line, and then the 3 class predicted secondary structure on a new line.
This doesn't adhere to the old 80 char line limit of FASTA files so it's easier to parse. 
If you'd like a quick parser script raise an issue or submit a pull request. 
The label is the Uniprot ID of the representative sequence of the Uniclust30 cluster that the example came from.

Importantly, this training set has had several different filters applied (see our paper) to remove homology from the CB513 test set.
This makes the dataset ideal for training not just secondary structure predictors but also unsupervised sequence models. 
In both cases, using this training set with CB513 as a test set provides a strong test of generalization.     

## Inference Code

If you'd like to train your own version of S4PRED using the AWD-GRU model we used we recommend building off the offical Salesforce 
AWD-LSTM (https://github.com/salesforce/awd-lstm-lm/) of which our model is a variant. The inference code used in this repo has been pared back to be clean and minimalist. As such, most of the things, like DropConnect, that make the AWD-LSTM what it is, are not present.  

### Batched Sequence Prediction

The current inference script produces predictions for sequences in batches of 1 for end-user clarity and is very fast regardless. That said, if you would like to run S4PRED on millions or billions of sequences there are obivously large benefits to be had from predicting in batches of several hundred sequences and higher. If this is of interest, please don't hesitate to raise an issue or get in touch. 

## Citation

If you use S4PRED in your work please cite the following link to the published article in Bioinformatics: 

**Increasing the Accuracy of Single Sequence Prediction Methods Using a Deep Semi-Supervised Learning Framework**</br>
Lewis Moffat and David T. Jones;
Bioinformatics, 07-2021,
DOI:10.1093/bioinformatics/btab491 
[LINK](https://doi.org/10.1093/bioinformatics/btab491)

Here is the corresponding BibTex: 

```bibtex
@article{10.1093/bioinformatics/btab491,
    author = {Moffat, Lewis and Jones, David T},
    title = "{Increasing the Accuracy of Single Sequence Prediction Methods Using a Deep Semi-Supervised Learning Framework}",
    journal = {Bioinformatics},
    year = {2021},
    month = {07},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab491},
    url = {https://doi.org/10.1093/bioinformatics/btab491},
}
```

## Changelog

For a log of recent changes please see the changelog in [CHANGELOG.md](https://github.com/psipred/s4pred/CHANGELOG.md). This is currently being updated manually by [@limitloss](https://github.com/limitloss).

  

## Contact

Current dev & maintainer is [@limitloss](https://github.com/limitloss). Please don't hesitate to reach out, either via my email, which can be found by clicking the [link to the paper](https://github.com/psipred/s4pred/CHANGELOG.md), and clicking on Lewis' name, or on twitter [@limitloss](https://twitter.com/limitloss).  

[changelog]: ./CHANGELOG.md
[license]: ./LICENSE
[version-badge]: https://img.shields.io/badge/version-1.1.0-green?style=flat-square&logo=appveyor.svg
[license-badge]: https://img.shields.io/badge/license-GPL--3.0-blue?style=flat-square&logo=appveyor.svg