### ImageEditor
A project to automatically edit images, and then local host a website that allows users to upload images

## Run Instuctions
To run the model the required pacages must first be installed through the command:
 - pip install requirments.txt

To train the a model the discriminator network needs to have an inital training. This is done through the script InitialDiscriminatorTraining.py. To train the generator and discriminator in training in tandem the script GeneratorTraining.py is used.

Both models take command arguments.
Below is a list of command args, the type they expect, their default value, and a description of the field.

--img_dir, type=str, default=os.getcwd(), Number of image pairs per batch.

--batch_size, type=int, default = None, Explicity declare minibatch size. Defaults to 25 per GPU otherwise.

--num_workers ,type=int,default = 2, Number of workers for retrieving images from the dataset. 

--epochs,type=int,default = 10000, Number of epochs to train for.

--no_cuda,type=bool,default = False, Flag to disable CUDA.

--number_images,type=int,default = 5000 # Number of images to use in train/test/val total.

--plot,type=bool,default = True, Flag to plot image examples each epoch.

--save_path,type = str, default = './'  Path so save generative model. 

--load_prev,type = bool, default = False, Load in a previous descriminator

--pretrained_discriminator_path, type=str, default='initial_discriminator_model_trained.pth', The path to the discriminator model.

Call the script with the argument specifier (--...), then the value desired for the argument to overwrite the default.

To deploy the model run the script image_editor_web.py. This launches a flask app which can be accessed through a browser tab using the url localhost:5000.

## Citations
Ignatov, Andrey, et al. “DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks.” ArXiv:1704.02470 [Cs], Sept. 2017. arXiv.org, http://arxiv.org/abs/1704.02470.

Vladimir Bychkovsky, Sylvain Paris, Eric Chan, & Frédo Durand (2011). Learning Photographic Global Tonal Adjustment with a Database of Input / Output Image Pairs. In The Twenty-Fourth IEEE Conference on Computer Vision and Pattern Recognition.

Computations were performed on the Vermont Advanced Computing Core supported in part by NSF award No. OAC-1827314.
