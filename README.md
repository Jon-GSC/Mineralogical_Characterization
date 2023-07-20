# Mineralogical_Characterization
* Machine learning approach to predict the mineral compositions using conventional and geochemical well logs

### Download link-note:

* The goal of this study is using a combined approach with previous U-Net and XGBoost algorithm for mineralization composition analysis using petrophysical logs from selected formationtrain. The code has been tested without any issue in PyCharm IDE. 

### Requirements: 

* Install python 3.8/newer and anaconda packages: conda install keras-2.4.3, tensorflow-2.3.1, scipy, scikit-image, tqdm, pandas, numpy, seaborn, shutil, matplotlib, lasio etc. libraries.


### Dataset: 

* The original dataset of conventional and geochemical logs are from Horn River Basin. The loaded data are noised-added and used for demonstration purpose only. 


### Instruction: 
     
   1. The three main python codes and _/itools_, _/data_ need be save in same folder. Before run the code, need install open-source Python packages.

   2. _forWeb1_Data_prepare.py_ is used to loading '.las' files, generate the training dataset, and save as '.pkl'.

   3. _forWeb2_Model_training.py_ is used to create the models, and training/validation with datasets from step 2. All of the weight parameters will be saved in one folder, code will run through each of the components and save all of the results in one folder.

   4. _forWeb3_Prediction.py_ is used for predicting of mineral composition with conventional petrophysical logs, and plotting figures.
  
   For pilot running, user does not need change any parameters, just run the code files in steps 2,3,4. The code should run without any error if environment setting is correct. Please contact at following email address if any bugs popup.


### Hardware tested: 

* HP-7920 workstation: 56core CPU; 64G memory; one Nvidia Quadro P5000 GPU.


### Contact: 

* jon.liu@nrcan-rncan.gc.ca


### Acknowledgments:

* This Mineralogical characterization study used open source codes and library from github, google, and open-sourced geoscience packages lasio. Please cite the related references in your publications.

![p_core_24971_edited_Github](https://github.com/Jon-GSC/Mineralogical_Characterization/assets/39324742/fda6d582-2002-4897-84aa-a474f1cb553e)

