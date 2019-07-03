# StreoScenNet
StreoScenNet: Surgical Stereo Robotic Scene segmentation
## Abstract
<p align="justify">
Surgical robot technology has revolutionized surgery toward a safer laparoscopic surgery and ideally been suited for surgeries requiring minimal invasiveness. Sematic segmentation from robot-assisted surgery videos is an essential task in many computer-assisted robotic surgical systems. Some of the applications include instrument detection, tracking and pose estimation. Usually, the left and right frames from the stereoscopic surgical instrument are used for semantic segmentation independently from each other. However, this approach is prone to poor segmentation since the stereo frames are not integrated for accurate estimation of the surgical scene. To cope with this problem, we proposed a multi encoder and single decoder convolutional neural network named StreoScenNet which exploits the left and right frames of the stereoscopic surgical system. The proposed architecture consists of multiple ResNet encoder blocks and a stacked convolutional decoder network connected with a novel sum-skip connection. The input to the network is a set of left and right frames and the output is a mask of the segmented regions for the left frame. It is trained end-to-end and the segmentation is achieved without the need of any pre- or post-processing. We compare the proposed architectures against state-of-the-art fully convolutional networks. We validate our methods using existing benchmark datasets that includes robotic instruments as well as anatomical objects and non-robotic surgical instruments. Compared with the previous instrument segmentation methods, our approach achieves a significant improved dice similarity coefficient.</p>

## Sample Result
![alt text](https://github.com/ahme0307/streoscene/blob/master/readme/image002.PNG)

## Network
![alt text](https://github.com/ahme0307/streoscene/blob/master/readme/fully2.png)

## How to train 
- To train the network with 4- fold validation 

> python R_Roboscene.py

- For intractive training and visualization checkout the jupyter notebook: <a href="https://github.com/ahme0307/streoscene/blob/master/Roboscene.ipynb">Roboscene.ipynb</a>  

## Reference
If you find this code useful please cite
>Mohammed, Ahmed, et al. "StreoScenNet: surgical stereo robotic scene segmentation." Medical Imaging 2019: Image-Guided Procedures, Robotic Interventions, and Modeling. Vol. 10951. International Society for Optics and Photonics, 2019.

@inproceedings{mohammed2019streoscennet,
  title={StreoScenNet: surgical stereo robotic scene segmentation},
  author={Mohammed, Ahmed and Yildirim, Sule and Farup, Ivar and Pedersen, Marius and Hovde, {\O}istein},
  booktitle={Medical Imaging 2019: Image-Guided Procedures, Robotic Interventions, and Modeling},
  volume={10951},
  pages={109510P},
  year={2019},
  organization={International Society for Optics and Photonics}
}
