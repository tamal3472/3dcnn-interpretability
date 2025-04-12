# Beyond Accuracy: Interpretability Methods in MRI-Based Diagnostic Models of Alzheimer’s Disease Detection


Some files are copied and forked from these two repositories: https://github.com/jrieke/cnn-interpretability, https://github.com/Nexer8/SE3D. We will be citing these two papers in our paper as well for their founding base.

    
    

## Code Structure

The codebase uses PyTorch and Jupyter notebooks. The main files for the paper are:

- `final_train_test_version-batch20.ipynb` & `final_train_test_version-instance20.ipynb` are the notebook to train and test the models. Additionally, they have the MRI preprocessing codes inside them.
- `final_interpretation_methods.ipynb` contains the code to generate all the interpretation methods. It also includes the code to reproduce all figures and tables from the paper.
- All `*.py` files contain methods that are imported in the notebooks above.




## Trained Model and Heatmaps

If you don't want to train the model you can use the uploaded trained models

- softmax-output_batch_normalization_e20.pt
- softmax-output_instance_normalization_final_e20.pt
- softmax-output_state-dict-batch_normalization_e20.pt
- softmax-output_state-dict-instance_normalization_final_e20.pt



## Data

The MRI scans used for training are from the [Alzheimer Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/). The data is free but you need to apply for access on http://adni.loni.usc.edu/. Once you have an account, go [here](http://adni.loni.usc.edu/data-samples/access-data/) and log in. 


### Tables

We included csv tables with metadata for all images we used in this repo (`data/ADNI CSV/ADNI_CN_AD.csv`). These tables were made by combining several data tables from ADNI. We used cross-sectional brainmasks



## Requirements

- Python 3
- Scientific packages (included with anaconda): numpy, scipy, matplotlib, pandas, jupyter, scikit-learn
- Other packages: tqdm, tabulate
- PyTorch: torch, torchvision 
- torchsample: Use the updated fork version for python 3 compatibility https://github.com/tamal3472/torchsample

