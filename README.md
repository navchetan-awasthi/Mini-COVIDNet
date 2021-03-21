# Mini-COVIDNet
Deep Neural Network for Ultrasound based Point-of-Care Detection of COVID-19

Mobile network based models are proposed for making smalle models for COVID-19 detection and 
compared with state of the art techniques for ultrasound imaging. We compared our models with other state
of the aret techniques such as POCOVID-Net and comapred our models in terms of size, number of parameters as well as the 
various figures of merit.

## Contributors : 
Navchetan Awasthi, and Phaneendra K. Yalavarthy

## Datasets :
The dataset utilized in this work is available at:
https://github.com/jannisborn/covid19_pocus_ultrasound/tree/master/data
Please use the same instructions to get the data for research purposes and do cite the relevant work.

## Models used for the comparisons
This repository conatins the notebooks for all the codes which were run for all the models proposed and compared.

The various models comapred here can be given as:

* COVID-CAPS : 

This architecture has been intially used for identification of COVID-19 from chest X-ray images. It consists if convoltuional layers, and capsule layers in the architecture.

* POCOVID-Net :

Here, a VGG-16 network architecture pretrained archietcture was used for the detection of COVID-19 for ultrasound images. 

* Mini-COVIDNet :

Here, a modified mobilenet architecture was used and shown to perform better for ultrasound images. It consists of depthwise convolution and separable convolution instead of normal convolution and shown to perform better. 

* Mini-COVIDNet (focal loss) :

Here, a modified mobilenet architecture was used and shown to perform better for ultrasound images. It consists of depthwise convolution and separable convolution instead of normal convolution and shown to perform better using the loss function involving the focal loss. 

* MOBILENetV2 :

Here, a modified mobilenetv2 architecture was used and shown to perform better for ultrasound images. It consists of depthwise convolution and separable convolution instead of normal convolution.

* NASNETMOBILE

Here, a modified NasNetMobile architecture was used and shown to perform better for ultrasound images. It consists of depthwise convolution and separable convolution instead of normal convolution and shown to perfrom better. 

