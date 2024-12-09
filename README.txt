CAGE-ASSISTED UNSUPERVISED OBJECT CLASSIFICATION: MULTI-MODEL PSEUDO LABELING APPROACH
Azeem Motiwala, Jay Choudhary and Chanakya Varude

We train the FastRCNN model on unlabelled data leveraging CAGE and SPEARs joint learning, with four labelling models. 

The FastRCNN folder contains the main files, where jl.py when run the joint learning will take place, etc.
The spear folder has some minor changes made to the library so as to adapt for our task, it does not contain the implyloss folder (this can be simply added by the github
https://github.com/decile-team/spear)
The labelling models are trained with the files resnet.py,vgg16.py,inception.py,resnext.py.
