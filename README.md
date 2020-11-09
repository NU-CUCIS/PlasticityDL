# Learning to Predict Crystal Plasticity at the Nanoscale:Deep Residual Networks and Size Effects in UniaxialCompression Discrete Dislocation Simulations
This software is an application of deep residual networks for predicting crystal plasticity at nanoscale. The efficacy of the proposed approach is tested on a crystal plasticity dataset. 

To use this software, what the algorithm requires as input are a numpy array. In this dataset, the shape of strain profile image is variaed, and it is converted to several (224, 224) crops after data preprocessing. The software will take the preprocessed image as input, and train the deep residual networks. The detailed drscription about data preprocessing and model can be found in the published paper given below.

## Requirements ##
* Python 3.6.3 
* Numpy 1.18.1 
* Sklearn 0.20.0 
* Keras 2.3.1 
* Pickle 4.0 
* TensorFlow 2.1.0 
* h5py 2.9.0
* Scipy 1.2.0

## Files ##
1. `model_training.py`: The script first applying data preprocessing on strain profile images, and train a deep residual networks to predict initial deformation levels. After training, the trained model will be saved in 'my_model.hdf5' file.
2. `data_file.pkl`: The file contains the file names of data in `image` folder as well as their labels.
4. `image` folder: Example data for crystal plasticity dataset, including 10 strain profiles.


## How to run it
1. Run commend below, which trains the deep residual network and save the trained model in 'my_model.hdf5' file.
   ```
   python model_training.py
   ```

## Acknowledgement
This work was performed under the following financial assistance award 70NANB19H005 from U.S. Department of Commerce, National Institute of Standards and Technology as part of the Center for Hierarchical Materials Design (CHiMaD). Partial support is also acknowledged from DOE awards DE-SC0014330, DE-SC0019358.

## Related Publications ##
Yang, Zijiang, Stefanos Papanikolaou, Andrew CE Reid, Wei-keng Liao, Alok N. Choudhary, Carelyn Campbell, and Ankit Agrawal. "Learning to predict crystal plasticity at the nanoscale: Deep Residual Networks and Size Effects in Uniaxial compression Discrete Dislocation Simulations." Scientific Reports 10, no. 1 (2020): 1-14.

## Contact
Zijiang Yang <zyz293@ece.northwestern.edu>; Ankit Agrawal <ankitag@ece.northwestern.edu>
