# Changelog

All notable changes to this project will be documented in this file.

This format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project would like to in spirit adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.6] - 2023-01-28

### Added

-  New ansible installation scripts in `ansible_installer/`.

## [1.2.5] - 2023-10-02

### Changed

- `README.md` to include download links for the real-labelled datasets.

## [1.2.4] - 2023-01-28

### Added

-  Option to supply a file prefix with a new command line option for `run_model.py`,

### Changed 

- `README.md` to reorder the About section, add a section to describe the new prefix option, and a section describing the new method's PSIPRED Workbench availability. 

## [1.2.3] - 2023-01-16

### Added

- Option to save the results in a secondary file format. E.g. a sequence prediction can be saved as both ss2 and horiz in the same run without having to run twice.  

### Changed 

- `README.md` to correct broken links and improve clarity.


## [1.2.2] - 2023-01-16

### Changed

- Fixed bug in `run_model.py` causing FASTA output to fail silently.
- Added single letter options for command line arguments.

## [1.2.1] - 2023-01-16

### Changed

- `README.md` to correct broken links and improve clarity.
- `s4pred.svg` from an inkscape SVG to a standard SVG so its not bloated with metadata and other unneccesary information. 


## [1.2.0] - 2023-01-06

### Added 

- Functionality to output files with the PSIPRED HFORMAT (`.horiz` files).
- An example of the output in `example/1qys.horiz`.
- Details to the `README.md` regarding the output format. 

### Changed

- Fixed version number of output ss2 file header in `run_model.py`.
- `README.md` to correct typos and improve clarity.

## [1.1.0] - 2023-01-04

### Added 

- Dependency on Biopython, specifically the SeqIO module.
- Example of a FASTA file containing multiple sequences, with `example/multi_seqs.fas`.
- Added line to README.md regarding the RCSB PDB no longer providing updated secondary structure flat files (i.e. `ss.txt.gz`).  
- Added descriptions of new output options.
- `example/1qys_ss_conf.fas` included as an example of adding confidence scores to the `.fas` format.
- `example/multi_seqs.fas` for examples of predicting from a file containing more than one sequence.

### Changed

- Behavior of `utilities.py` and `run_model.py` so they now parse and predict multiple sequences from a single FASTA file. 
- `utilities.py` uses a new wrapper around the Biopython parser.
- `run_model.py` refactored to predict from multiple sequences in a single FASTA files. 

### To Do

- [ ] Add name and ID post-processing after Biopython read to improve the downstream file naming.
- [ ] Add separate inference script for large dataset processing with batches.


## [1.0.0] - 2023-01-03

### Added 

- README section on the inference code.

### Changed

- Corrected invalid username references in python files.

## [0.0.1] - 2023-01-03

### Added 

- This CHANGELOG file.
- README now contains references and descriptions of:
-- CHANGELOG. 
-- Contact for main developer [@limitloss](https://github.com/limitloss). 

### Changed

- Slimmed down the contents of the Citation section in the README. 
- Fixed badges and made them visible.
- Widened the banner SVG file, and the corresponding PNG banner image file.

### Removed

- Removed hench deleted files from main branch's history, drastically reducing the pack-file size.
