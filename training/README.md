


# Augmenting Data
We had two main augmentation files  ``augment.ipynb`` and ``augment_attempt_2.ipynb`` 
The first of which, ``augment.ipynb`` is to be used on images that have been split from the default camera output of left and right
This file will flip, blur, and change the brightness of an image if the image belongs to a turn. This first attempt at augmentation was 
a way of dealing with extremely unbalanced data, as we were driving in a completely straight path at this point. This helped balance the data, 
but a slight drift would cause our model to have no chance of recovering. Currently, we have some of these augmented images stored in output_folder.

Our next augmentation file ``augment_attempt_2.ipynb`` was a supplement to our data collection methodology. After driving the S pattern, we saw that there were 
still many flaws that caused our robot to crash into the wall. While we were gathering more natural turn data, we were teaching the robot incorrect behavior. This is because during the S pattern, we would be turning into the wall, which was teaching the robot that turns into the wall was good. Instead, we wanted to teach it to turn the other way when looking at a wall, and to do this, we would have to negate the velocity while we were collecting data of the robot turning into the wall. So we decided to still drive the S pattern, but be able to recognize when the robot was turning towards the wall. To do this, we had an angular velocity (turn velocity) of magnitude 0.3 while driving towards the wall, and a magnitude of 0.8 when turning away from the wall. We were able to use this to negate the velocity while turning into walls, to tell our model to correct away from the wall rather than going toward the wall. This worked well in corridors, except at turns. The amount of turn data was negligible when compared to the overall amount of data, so the robot didn't learn to make turns well. To fix this issue, we implemented a trigger that would store a boolean indicating that our robot was doing a "real" turn rather than the S pattern. This allowed us to augment/multiply these turn images to help the model better learn these situations. 

Running these augmentation files will require you to:
1. Change the path to the dataset that you wish to augment
2. Change the name of the .csv file to data.csv, or change it in the path name
3. Set the path of where you want to save the augmented dataset. 

We also have an image_splitting.py file in our root folder since our robot captures slightly translated images from two cameras (left and right) and one might want to use those as separate images. However, we are not making use of this code currently because it might introduce errors while augmenting the images (particularly while flipping).

# DAVE2 Model Setup
Ensure that the ``train.sh`` points to the dataset parent directory of the dataset you want to train on. 
To install the dependencies required for training the model, you can run ``pip install -r requirements.txt``. Alternatively, you can also run the ``install.sh`` file to install all the dependencies. In case a package is unable to be found, you can manually install it using ``pip install <package_name>``. The model is designed to take input images of shape (3, 640, 360), where 3 represents the number of channels (RGB), and 640x360 is the spatial resolution.

The DAVE2 model architecture is stored under /models/DAVE2pytorch.py. In particular, we are using the DAVE2V3 class for training our model.  

The architecture consists of convolutional layers followed by max-pooling layers to extract hierarchical features from the input image. The flattened output is then processed through fully connected layers, incorporating dropout for regularization. The model is designed for a regression task with a single output node.

More information about the DAVE2 architecture can be found here: [Nvidia's DAVE-2 system](https://arxiv.org/pdf/1604.07316v1.pdf)

# Training a DAVE2 model
This portion of the code assumes that you have collected/augmented your data. You should have a directory that is for your datasets. Ours was named ``/datasets``. You can have as many datasets as you wish in this folder. The DatasetGenerator.py file should read and combine all of the datasets. Our overall dataset is split into a training set and a validation/test set, with 80% of the data being in the training set. This helped us see when our model was overfitting and when extra epochs were not helping the model anymore. Originally we were using the robustification part of the DatasetGenerator.py, but we stopped using that because that would modify images in place, and in most cases, we wanted to increase the number of images, rather than just changing them. This led to us creating these augmentation files. 

To summarize the steps to train your own DAVE2 Model:
1. Have data in a /datasets directory
2. Change the paths in train_DAVE2 to where you want to store the best model weights
3. Run ``train.sh`` or ``python3 train_DAVE2.py #PATH OF DATASET`` (If you're using Windows, you need to use ``bash train.sh``. You may also need to use the dos2unix command before running the .sh files).
4. Grab the best model from your folder after the training is finished and load it into the steering_NN

In case you're using UVA's CS department SLURM, you can follow these steps in place of the 3rd one above:
1. Log into the department portal nodes: ``ssh computing-id@portal.cs.virginia.edu``
2. Clone this repo (or your fork of this repo) into your home directory: ``git clone git@github.com:ish-gupta/ml-robot.git``
3. Navigate to the training directory: cd ~/ROSbot_data_collection/training
4. Create a Python virtual environment and install requirements using the script provided: ./install.sh.
5. Copy your datasets to the ``datasets`` directory
6. Edit the ``train.sh`` script to point to the dataset parent directory of the dataset you want to train on.
7. Check what slurm GPU nodes are available using the ``sinfo`` command. Nodes marked "idle" mean they are available for you to launch jobs on them.
8. Launch the job on slurm using a configuration that looks like below. You can replace the ``adriatic05`` with whichever GPU node is available to train on. ``--gres=gpu:1`` specifies the generic resource (gres) requirement for the job. In our case, it requests one GPU. ``--exclusive=user train.sh``indicates that you want to request exclusive access to resources for the user running the job.

```
sbatch -w adriatic05 -p gpu --gres=gpu:1 --exclusive=user train.sh
```
 
9. Check the job periodically to be sure it is progressing using the ``squeue -u $USER`` command, and check the log according to the ``$SLURM_JOB_ID`` using ``cat slurm-JOB_ID.out``.


If running on a local machine, you may want to run ``caffeinate python3 train_DAVE2.py #PATH OF DATASET``, as this will keep your machine up and running even if the screen turns off. Another suggestion is that if your epochs are set to be too high, your "best" model might be too overfit, so you can take the best model at any epoch while the model is training. 

# Model Layers

### Input Layer
The model takes in input images with a shape of (3, 640, 360), where 3 represents the RGB channels, and 640x360 is the spatial resolution.

### Convolutional and MaxPooling Layers
Three convolutional layers (conv1, conv2, conv3) with increasing output channels (16, 32, 64) are applied to extract hierarchical features from the input image. Between each convolutional layer, max-pooling layers (pool1, pool2, pool3) are used to downsample the spatial dimensions while preserving important features.

### Flattening
The output from the convolutional and pooling layers is flattened using nn.Sequential and np.prod. This flattened representation is used to determine the size of the subsequent fully connected layers.

### Fully Connected Layers with Dropout
Three fully connected (linear) layers (lin1, lin2, lin3) are employed for further feature processing. Dropout is applied after the first two fully connected layers (dropout1, dropout2) with dropout probabilities of 0.5 and 0.25, respectively. Dropout helps prevent overfitting by randomly "dropping out" neurons during training.

### Output Layer
The final linear layer (lin4) produces a single output feature since the model is designed for a regression task.

The model's weights are initialized using a custom weight initialization method (init_weights) that initializes the weights of the model layers based on the Xavier (Glorot) uniform initialization for weights and zero initialization for biases.

# Hyperparameters
We defined some default hyperparameters in our code using the parse_arguments function:
1. Batch Size (``batch``): A batch size of 64 is specified by default, and chosen based on the available memory resources. We think this is the optimal batch size for our case that can provide a regularization effect and allow the model to generalize better.
2. Number of Epochs (``epochs``): A default value of 40 is used which is decided based on the convergence behavior observed during training. If convergence is observed earlier, we can use the .pt files that are generated after every epoch.
3. Learning Rate (``lr``): The learning rate is set to 1e-3, a common starting point for many optimization tasks. This value is neither too high, risking convergence issues, nor too low, potentially causing slow convergence.
4. Log Interval (``log_interval``): It is set to 50, indicating that training information is logged every 50 batches. This parameter helps in monitoring the training process and can be adjusted based on the desire for more or less frequent updates.
