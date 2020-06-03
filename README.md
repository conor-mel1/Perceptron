# Perceptron

This repository contains code for a Perceptron classification algorithm created as part of Assignment 3 of the CT475 Module - Machine Learning and Data Mining, at NUI Galway.

The implementation of the algorithm is "from scratch" as per the assignment specifications. The purpose of the classification algorithm is to distinguish between three types of owl: Barn Owl, Snowy Owl, and Long-Eared Owl.

The constructed perceptron is not capable of multi-class classifications and so the original dataset is modified to produce 3 separate CSV files, with a one verses the rest approach adopted.
For example, the perceptron tries to detect the Snowy Owl versus other owls, Barn Owl versus other owls etc. 

The data used is contained in the data folder, with the src folder containing the Perceptron code.
Instructions on how to run the code are contained within the code.