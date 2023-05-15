# Identifying the Effects of Russian Aggression on Agricultural Fields in  Ukraine through Classification  Approaches and Satellite Imagery

## Contents

### [tiff_utils](https://github.com/sophmintaii/MineFree/tree/main/tiff_utils)

#### [cut_tiff.py](https://github.com/sophmintaii/MineFree/blob/main/tiff_utils/cut_tiff.py)

This script is used for cutting the TIFF images into patches of the given size.
It contains function for just splitting the TIFF image into patches, as well as splitting
and assigning them a 'bombed' ot 'not-bombed' label based on the contents of the 
segmentation mask. Usage of the latter function:
```bash
usage: cut_tiff.py [-h] [--input INPUT] [--mask [MASK]] [--output OUTPUT] [--size SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to the input tiff file.
  --mask [MASK], -m [MASK]
                        Path to the mask tiff file.
  --output OUTPUT, -o OUTPUT
                        Path to the folder with output images.
  --size SIZE, -s SIZE  Size of the output images.
```

#### [join_tiff.py](https://github.com/sophmintaii/MineFree/blob/main/tiff_utils/join_tiff.py)
This script was used to join the segmentation masks together, as the umages are split into smaller ones
before being passed to the annotators for convenience.
```bash
usage: join_tiff.py [-h] [--input INPUT] [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to the input directory.
  --output OUTPUT, -o OUTPUT
                        Path to the output directory.
```

### [hyperparams_search.ipynb](https://github.com/sophmintaii/MineFree/blob/main/hyperparams_search.ipynb)

This Jupyter notebook that was exported from the Google Colab is used for the hyperparameters search.

It contains the whole pipeline from the definition of the essential classes such as ```Classification_Task```, ```DataModule```, etc. 
to the running the WandB sweep.

To run this notebook, you would need to log in with your own Weights&Biases credentials and paste your ```entity-name``` and ```project-name``` into the according notebook cell.