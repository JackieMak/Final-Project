# Final-Project
Title:
Predicting the confident value of amino acid in protein by using 3D Convolutional neural network (3DCNN)

Description:
Proteins are made up of chains of amino acids which fold into unique 3D structures to give each
protein a different function. Google DeepMind’s AlphaFold can predict the structure of protein with
apparent high accuracy which can help scientists to understand how the protein works and what the
protein function is. This information might help solve the most significant challenges in the world,
such as developing treatments for diseases or finding enzymes that break down industrial waste.
AlphaFold is sometimes unsure about its prediction of part of a structure and understanding this uncertainty
is important for improving protein structure prediction. This project will endeavour to learn the behaviour
of AlphaFold on uncertain areas of the protein. Specifically, this project aims to predict the confidence value
of residues in a protein provided by AlphaFold given the local atomic neighbourhood. 3D convolutional neural network (3DCNN)
will be the method that will be used in this project. First, a 20x20x20 angstrom box will be created and
positioned at each residue’s alpha-carbon atom. Second, the number of different element atoms that appear
in the voxels will be counted. Lastly, a 3DCNN will be used to predict the amino acid’s confidence value using the voxel data as input.

•	Software Resources
o	IDE: JetBrain PyCharm Community (Version: 2021.2.3), in both Windows and macOS Apple Silicon version.
o	Environment: Python 3.8 – Anaconda
o	Packages:
	NumPy v1.19.5
	Torch v1.10.0
	Biopandas v0.2.9

•	Hardware Resources
o	Alienware 17 R4 with Windows 10 system
o	MacBook Pro 13’’ with Apple M1 chip
