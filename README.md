neNequitia
==========

neNequitia is a software aimed at evaluating CER without ground-truth,
to help design transcription campaign in HTR data creation campaigns.
By providing insight on the estimated results of models, users can
focus on seemingly badly transcribed manuscripts or improve the medium results.

## Cite

Use the CITATION.cff or 

```bibtex
@inproceedings{clerice:hal-03828529,
  TITLE = {{Ground-truth Free Evaluation of HTR on Old French and Latin Medieval Literary Manuscripts}},
  AUTHOR = {Cl{\'e}rice, Thibault},
  URL = {https://hal-enc.archives-ouvertes.fr/hal-03828529},
  BOOKTITLE = {{Computational Humanities Research Conference (CHR) 2022}},
  ADDRESS = {Antwerp, Belgium},
  YEAR = {2022},
  MONTH = Dec,
  KEYWORDS = {HTR ; OCR Quality Evaluation ; Historical languages ; Spelling Variation},
  PDF = {https://hal-enc.archives-ouvertes.fr/hal-03828529/file/CHR2022___State_of_HTR.pdf},
  HAL_ID = {hal-03828529},
  HAL_VERSION = {v1},
}
```

## Install

Use `pip instal -r requirements`


## Structure

- Jupyter notebook models are used for analyzing and running experiments.
- The `nenequitia` module is a stand-alone module for development.

## Data

Most of the data and models for the paper are available on the release page ( https://github.com/PonteIneptique/neNequitia/releases/tag/chr2022-release )

The list of manuscripts, their automatic transcription with the best model, the full ground truth in XML format of the paper and the predictions of NeNequitia for the automatic transcription of the manuscripts are to be found here : https://zenodo.org/record/7234399#.Y1-d_L7MJhE


## License

Mozilla Public License 2.0