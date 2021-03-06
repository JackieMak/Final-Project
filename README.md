# Final-Project
Predicting the confident value of amino acid in protein by using 3D Convolutional neural network (3DCNN)
## Introduction
The project concept is based on the AlphaFold Artificial Intelligence (AI) technology (Jumper et al., 2021).

Proteins are made up of chains of amino acids which fold into unique 3D structures to give each protein a different function. Google DeepMind’s AlphaFold can
predict the structure of the protein with apparent high accuracy, which can help scientists understand how the protein works and what the protein function is. This
information might help solve the most significant challenges in the world, such as developing treatments for diseases or finding enzymes that break down industrial
waste. AlphaFold is sometimes unsure about its prediction of part of a structure, and understanding this uncertainty is essential for improving protein structure 
prediction. This project will endeavour to learn the behaviour of AlphaFold on uncertain areas of the protein. Specifically, this project aims to predict the 
confidence value of residues in a protein provided by AlphaFold given the local atomic neighbourhood.

3D convolutional neural network (3DCNN) will be the method that will be used in this project. First, a 20×20×20 angstrom box will be created and positioned at each
residue’s alpha-carbon atom. Second, the number of different element atoms that appear in the voxels will be counted. Lastly, a 3DCNN will be used to predict the 
amino acid’s confidence value using the voxel data as input. 

## Software & Hardware Resources
### Software Resources
#### Environment: 
Python 3.8 – Anaconda
#### IDE: 
Google Colab

JetBrain PyCharm Community (Version: 2021.2.3), in both Windows and macOS Apple Silicon version.
#### Packages:
Biopandas v0.2.9, Matplotlib v3.4.2, NumPy v1.19.5, Scikit-learn v0.24.2, Torch v1.10.0
### Hardware Resources
Alienware 17 R4 with Windows 10 system

MacBook Pro 13’’ with Apple M1 chip
