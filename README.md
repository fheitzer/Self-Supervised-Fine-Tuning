# Self-Supervised-Fine-Tuning
This is where we finely tune an ensemble self supervised

## Datasets
Load them and put in data folder, each class 1 subfolder pls.
Adjust ensemble num_outputs accordingly.
Metadata for MSK, UDA, HAM10K, PH2, DERM7PT, BCN20k is provided.
Here only histopathologically labelled data was used, and resized to 600 x 450, to match the dataset with the lowest resolution (ham10k)

### ISIC-CLI
Create isic api account
´´´
https://api.isic-archive.com/images/
pip install isic cli
cli user login
E.g. BCN only melanomas
isic image download --search "diagnosis:melanoma" --collections "249" --limit 0 myimages/
´´´

## Data Transformations
Necessary as there is not much data to work with
- Random Horizontal and Vertical Flip
- Random Resize and Crop 0.8, 1.2
- Color Jitter
  - brightness 64
  - contrast 0.75
  - saturation 0.25
  - hue 0.04
  - normalize

## Custom DataHandler
to yield paths for collection.

## Duplicates
Duplicates (MSE == 0) were deleted.
BCN mel: 47
BCN nv: 47

Many more similar found (MSE <= 50 && >0). Photos of the same lesion with a slightly different angle or crop. Not deleted.
BCN mel: 751
BCN nv: 751


## Set up
### Environment
python=3.10

pytorch 2, timm, torchvision, torchlightning

pip install torch==2.0.1+cu118 torchvision -f https://download.pytorch.org/whl/torch_stable.html


## Outlook
