# satellite-cloud-removal-dip
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cidcom/satellite-cloud-removal-dip/blob/main/01-Example-Use-colab.ipynb) [![DOI:10.1007/978-3-319-76207-4_15](https://zenodo.org/badge/DOI/10.3390/rs14061342.svg)](https://doi.org/10.3390/rs14061342)

Official code for the paper (DOI and official links will be posted soon):
```bibtex
@article{rs14061342,
author = {Czerkawski, Mikolaj and Upadhyay, Priti and Davison, Christopher and Werkmeister, Astrid and Cardona, Javier and Atkinson, Robert and Michie, Craig and Andonovic, Ivan and Macdonald, Malcolm and Tachtatzis, Christos},
title = {Deep Internal Learning for Inpainting of Cloud-Affected Regions in Satellite Imagery},
journal = {Remote Sensing},
volume = {14},
year = {2022},
number = {6},
article-number = {1342},
url = {https://www.mdpi.com/2072-4292/14/6/1342},
ISSN = {2072-4292},
DOI = {10.3390/rs14061342}
}
```
Please cite accordingly if any part of the repository is used.

## :cloud: Removing clouds with Deep Image Prior
[Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior) can be used for inpainting image areas in an internal fashion, without requiring anything other than the image itself and **the mask of the inpainted region**.

Furthermore, you can easily use any other collocated sources of information, like historical optical image without clouds, or a SAR optical image. In the example notebook `01-Example-Use`, we show how to do it with example data that includes two extra sources of information.

![Example Result](example_result.png?raw=true "Title")

## :computer: Implementation
This is all you need to do to set up the model:
```python
my_model = LitDIP()

my_model.set_target([s2_image, s2_mean, s1_image])
my_model.set_mask([mask,
                   np.ones(mask.shape),
                   np.ones(mask.shape)])
```
`LitDIP()` is a pytorch lightning wrapper for our model, that builds on top of the basic lightning functionality.

You can use `.set_target()` method and provide as many images as you need in a list. In this case, we supplied the `s2_image`, which we wish to inpaint, and two informing sources, `s2_mean` and `s1_image`.

The masks can be set using `.set_mask()` with a list of masks corresponding to the targets in `.set_target()`. If all images are of size 256x256xC, then a mask should have the shape 256x256.

To train, all you need to do is:
```python
trainer = pl.Trainer(
    max_epochs = 4,            
    checkpoint_callback=False,
    logger=False,
    gpus = [0]
)
trainer.fit(my_model)
```

Finally, to get the output of the model, use `.output()`:
```
result, _, _ = my_model.output()
```
The `.output()` method returns all reconstructed sources, so pay attention to the order. In this case, `s2_image` was the first target out of three images supplied to `.set_target()`, so result is the first out of three returned arrays.

## :wrench: Data format
So, what is the required data format? The nice feature of a DIP-based solution is that you're free to use almost any format and scaling of your data.

### Image Shape
It's fully convolutional so it will for for many spatial dimensions. Some shapes may be more problematic due to downsampling-upsampling inconsistencies, and generally, shape sizes based on powers of 2 work best. In the example, we use a size of 256x256.

### Value scaling
Any value range is fine. By default, `sigmoid_output = True` for `LitDIP()`, so you want to change it to `False` if the network has to produce values outside of `[0,1]`.

## :open_file_folder: Supporting Dataset
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5897694.svg)](https://doi.org/10.5281/zenodo.5897694)

You can download directly using
```bash
wget https://zenodo.org/record/5897694/files/dataset-for-zenodo.zip
```
The dataset used in the experiments presented in the manuscript can be found here:
https://doi.org/10.5281/zenodo.5897694

It contains two years of coverage (2019 and 2020) for two distant geographical areas in India and in Scotland.

## :clipboard: Requirements
These are the crucial packages for the project:
```bash
pytorch-lightning==1.2.0
torch==1.8.1
numpy=1.19.2
rasterio=1.0.21
```

> Please feel free to post any issues via GitHub, or pass them directly to mikolaj.czerkawski@strath.ac.uk
