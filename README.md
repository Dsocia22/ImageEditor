### ImageEditor
A project to automatically edit images, and then local host a website that allows users to upload images

## Run Instuctions
To run the model the required pacages must first be installed through the command:
 - pip install requirments.txt

To train the a model the discriminator network needs to have an inital training. This is done through the script InitialDiscriminatorTraining.py. To train the generator and discriminator in training in tandem the script GeneratorTraining.py is used.

To deploy the model run the script image_editor_web.py. This launches a flask app which can be accessed through a browser tab using the url localhost:5000.

## Citations
Ignatov, Andrey, et al. “DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks.” ArXiv:1704.02470 [Cs], Sept. 2017. arXiv.org, http://arxiv.org/abs/1704.02470.

Vladimir Bychkovsky, Sylvain Paris, Eric Chan, & Frédo Durand (2011). Learning Photographic Global Tonal Adjustment with a Database of Input / Output Image Pairs. In The Twenty-Fourth IEEE Conference on Computer Vision and Pattern Recognition.

Computations were performed on the Vermont Advanced Computing Core supported in part by NSF award No. OAC-1827314.
