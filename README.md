## JDAP(**J**oint **D**etection and **A**ilgnment and Head **P**ose)
A face detection algorithm joint multi-task using cascade structure in CNN. This repository containtraining code and testing code using Tensorflow architecture.

### Results

### Requirement
Quick overview of requirements:
    - Linux(ubuntu 16.04 my test)
	- tensorflow 1.0.1(or more high)
	- python 2.7
	- opencv-python
	- easydict

### Preparation Data
	- WIDER-FACE
	- 300W-LP + Menpo-3D or (CelebA + AFLW)

### Data Label Design

#### Classification task:
Positive, Part, Negative sample label : image_name class_id bounding_box_regressor
class_id: +1(Positive) -1(Part) 0(Negative)
Negative bbox_regressor: 0 0 0 0(anything, but must keep 4 number)
bounding_box_regressor: x1 y1 x2 y2(relative ground truth)

#### Auxiliary task:
Landmark sample label: image_name class_id landmark_regressor
Pose sample label: image_name class_id pose_regressor
class_id: -2(Landmark) -3(Pose)

### Training PNet
1. Modfiy **data_root_dir** and **save_dir_root** of "prepare_data/gen_pnet_train_data.py" and "prepare_data/gen_pnet_val_data.py" .
2. Select suitable parameters. 
- **Main options:**
  - IoU thresh
  - how many negative samples per image
  - pos_aug_ratio(without augment model performance is good. T^T)
3. > ./scripts/make_pnet_train_val_data.sh
4. Modfiy "prepare_data/gen_imglist.py" **save_data_dir **and **netSize**
5. According to self task design label in "prepare_data/multithread_create_tfrecords.py" **_set_single_example**
6. Modify "scripts/make_tfrecords.sh" and run
7. Adjustment "scripts/train_cls.sh" and run

### Training RNet
1. Modfiy "prepare_data/gen_hard_sample.py" set yourself fold root
2. **stage = 1** get PNet detect result and save it as pickle
3. **stage = 2** crop and save image patch
4. Like PNet generate tfrecords
5. Adjustment "scripts/train_cls.sh" and run

### Training ONet
1. Modfiy "prepare_data/gen_hard_sample.py" set yourself fold root, get classify samples
2. 

### Demo
Select model file and suitable hyper parameters
> python ./demo/mtcnn.py

### FAQ
1. lr = 0.01, lr_decay_scale = 0.1 and epoch [7 ,13] make lr decay
	
2. Small batch size(BS)
	- Small BS can get higher recall than large BS in FDDB.
3. l2 regularizer is small
	- Change normal 5e^-5 to 1e^-5, network less limit.
4. Add part samples in trianing stage
	- Help to bounding box regression and indirect promote face classification.
5. Less channel
	- There's negligible advance using more channel.
6. Optimizer
	- Momentum optimizer in 0.9 momentum.
7. OHEM achieve, ohem ratio is 0.7
	- BP top 70% of loss. The loss except part samples.
8. Focal loss VS. SoftmaxWithLoss
	- SF is more less false positive and higher recall than FL.
9. ERC(Early recject classifier) and DR Layer
	- New strategy in [2], but increase more parameters.

### Future work
1. PNet use conv1-s2 replace max pooling and relu6 replace prelu. But relu6 and reduce max pooling method lead
to more false positive ratio and lower recall.



### References
1. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks. Signal Processing Letters, 23(10):1499–1503, 2016.
2. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks. Signal Processing Letters, 23(10):1499–1503, 2016.
3. V. Jain and E. Learned-Miller. FDDB: A benchmark for face detection in unconstrained settings. In Technical Report UMCS-2010-009, 2010.
4. S. Yang, P. Luo, C.-C. Loy, and X. Tang. Wider face: A face detection benchmark. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
