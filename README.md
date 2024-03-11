# PA_AttackPlane
Physical adversarial patch-based attack for arerial airplane image object detection

## Introduction
We use a patch-based physical adversarial attack framework [AP-PA]([https://markdown.com.cn](https://github.com/JiaweiLian/AP-PA)),which aims to accomplish effective adversarial attack for aerial images containing airplane objects.
We successfully attack both one-stage and two-stage detection network,that is YOLOv5 and Faster RCNN.And the adversarial patch could adaptively resize according to the object size.In addition to the original project code,our project accomplished the patches locate out of airplane objects and only half of the objects are applied the adversarial patches,which prompt the convenience for use.

An example for our adversarial as follows:(images on the top row shows the appropriate results without attacks,and the bottom row shows the detect results after adversarial attacks)
![1.png](https://img2.imgtp.com/2024/03/11/ESgYed5c.png)
The P-R Curve before and after adversatial attack are as follows:(images on the top row shows the P-R curve before and after attacks on YOLOv5,and the bottom row shows the P-R curve before and after attacks on Faster RCNN)
![2.png](https://img2.imgtp.com/2024/03/11/H3cTV1r6.png)
The process of adversarial patches' changes is shown below:
![3.png](https://img2.imgtp.com/2024/03/11/iCUPpoz7.png)
And the applied image is shown below:(adversarial patches locate out of the airplane and only half of the planes are applied patches)
![epoch_2.jpg](https://img2.imgtp.com/2024/03/11/5oEtaSBa.jpg)

## Acknowledgement
This repository was my term project of AI built on top of [AP-PA]([https://markdown.com.cn](https://github.com/JiaweiLian/AP-PA)). We thank the effort from our community.
Zack
