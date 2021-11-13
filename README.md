![S4PRED Logo](./images/s4pred.png)
 
# S4PRED

A tool for accurate prediction of a protein's secondary structure from only it's amino acid sequence. 

This repo contains the fully trained inference model built in Pytorch. Its a simple one line command to run it on a fasta file containing a protein sequence. S4PRED is a product of our paper [Increasing the accuracy of single sequence prediction methods using a deep semi-supervised learning framework](https://academic.oup.com/bioinformatics/article/37/21/3744/6313164) published in Bioinformatics.

We also provide the 1.08M example pseudo-labelled training set used for training S4PRED. This has been cross-validated against the CB513 test set using a variety of sequence-searching-based methods, as real structure labels aren't available. Proper cross-validation when working with large sets of protein sequences is incredibly important, especially when working with powerful parametric models like deep neural networks.

S4PRED is a state-of-the-art single-sequence model meaning it doesn't use homology information to make predictions, only the primary amino acid sequence. 



# Requirements

- Python 3.7 or greater
- Pytorch 1.5.1 

The script hasn't been tested on newer Pytorch builds but it should run without any issues. 

# Installation

Clone the directory
```bash
git clone https://github.com/psipred/s4pred
cd s4pred
```
Make the `weights/` directory and `cd` into it, then download the model weights from our public server.
```bash
mkdir weights
cd weights
wget http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights_1.pt
wget http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights_2.pt
wget http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights_3.pt
wget http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights_4.pt
wget http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights_5.pt
``` 
Each weight file is ~86MB and so all together make up roughly 430MB. If you have python and Pytorch installed you should be ready to go. 

# Usage

The S4PRED model can be used to predict a sequence's secondary structure with the following:
```bash
python run_model.py YOUR_FASTA.fas > YOUR_OUTPUT.ss2
```
The results of the prediction are piped to `stdout` and prediction should take less than a second. 
There are two optional arguments you can give:

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

The alternative `fas` output returns the sequence FASTA file with the predicted secondary structure concatenated on a second line. This is similar to the FASTA flat-file that the RCSB PDB provides for all sequences and their DSSP based secondary structure (Downloaded from [here](https://cdn.rcsb.org/etl/kabschSander/ss.txt.gz)). Here is an example:
```
>1QYS_1|Chain A|TOP7|Computationally Designed Sequence
MGDIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQL
CCCEEEEEEECCCCCEEEEEEEECCHHHHHHHHHHHHHHHHHCCCCEEEEEEEECCHHHHHHHHHHHHHHHHHCCCCEEEEEEECCEEEEEEEC
```
The above example output of this file is located in `examples/1qys_ss.fas`.

## Example Run
The following is an example run on the sequence of TOP7 (PDB ID: 1QYS) using the GPU and output to the FASTA like format. The corresponding fasta input file is located in `examples/1qys.fas` (this is the PDB FASTA file stripped of the 6-HIS tag on the C-Terminus). 
```bash
python run_model.py --device gpu --outfmt fas example/1qys.fas > 1qys_ss.fas
```


# Dataset
We have made the pseudo-labelled training set available to download from our public server. 
These are in a simple FASTA flat file, `s4pred_train.fas`.
```bash
wget http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/s4pred_train.fas
```
This now available! 

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

# Citation

If you use S4PRED in your work please cite the following link to the published article in Bioinformatics: 

**Increasing the Accuracy of Single Sequence Prediction Methods Using a Deep Semi-Supervised Learning Framework**</br>
Lewis Moffat and David T. Jones;
Bioinformatics, 07-2021,
DOI:10.1093/bioinformatics/btab491 
[LINK](https://doi.org/10.1093/bioinformatics/btab491)

Here is the corresponding BibTex 

```bibtex
@article{10.1093/bioinformatics/btab491,
    author = {Moffat, Lewis and Jones, David T},
    title = "{Increasing the Accuracy of Single Sequence Prediction Methods Using a Deep Semi-Supervised Learning Framework}",
    journal = {Bioinformatics},
    year = {2021},
    month = {07},
    abstract = "{Over the past 50 years, our ability to model protein sequences with evolutionary information has progressed in leaps and bounds. However, even with the latest deep learning methods, the modelling of a critically important class of proteins, single orphan sequences, remains unsolved.By taking a bioinformatics approach to semi-supervised machine learning, we develop Profile Augmentation of Single Sequences (PASS), a simple but powerful framework for building accurate single-sequence methods. To demonstrate the effectiveness of PASS we apply it to the mature field of secondary structure prediction. In doing so we develop S4PRED, the successor to the open-source PSIPRED-Single method, which achieves an unprecedented Q3 score of 75.3\\% on the standard CB513 test. PASS provides a blueprint for the development of a new generation of predictive methods, advancing our ability to model individual protein sequences.The S4PRED model is available as open source software on the PSIPRED GitHub repository (https://github.com/psipred/s4pred), along with documentation. It will also be provided as a part of the PSIPRED web service (http://bioinf.cs.ucl.ac.uk/psipred/)Supplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab491},
    url = {https://doi.org/10.1093/bioinformatics/btab491},
    note = {btab491},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btab491/38853041/btab491.pdf},
}
```




