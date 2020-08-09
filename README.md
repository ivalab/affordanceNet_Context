# AffordanceNet_Context
This is the implementation of our submission 'Improving Affordance Detection on Novel Objects with Regional Attention toward Real-world Robotic Manipulation'. This paper presents a framework to apply attention and attribute on region-based architecture for affordance detection on novel objects to assist with robotic manipulation tasks. The original arxiv paper can be found [here](https://arxiv.org/pdf/1909.05770.pdf).

<p align="center">
<img src="https://github.com/ivalab/affordanceNet_Context/blob/master/fig/concept_plot_pddl.png" alt="drawing" width="300"/>
</p>

------------------------------------

### Requirements

1. Caffe:
	- Install Caffe: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html).
	- Caffe must be built with support for Python layers.

2. Specifications:
	- CuDNN-5.1.10
	- CUDA-8.0


### Demo

1. Clone the AffordanceNet_Context repository into your `$AffordanceNet_Context_ROOT` folder
```
git clone https://github.com/ivalab/affordanceNet_Context.git
cd affordanceNet_Context
```

2. Export pycaffe path
```
`export PYTHONPATH=$AffordanceNet_Context_ROOT/caffe-affordance-net/python:$PYTHONPATH`
```

2. Build Cython modules
```
cd $AffordanceNet_Context_ROOT/lib
make clean
make
cd ..
```

4. Download pretrained models
    - trained model for DEMO on [dropbox](https://www.dropbox.com/s/4wai7v9j6jp7pge/vgg16_faster_rcnn_iter_110000_pam_7attribute.caffemodel?dl=0) 
    - put under `./pretrained/`

5. Demo
```
cd $AffordanceNet_Context_ROOT/tools
python demo_img.py
```
	
### Training
1. We train AffordanceNet_Context on UMD dataset
	- You will need synthetic data and real data in Pascal dataset format. 
	- For your convinience, we did it for you. Just download this file on [dropbox](https://www.dropbox.com/s/zfgn3jo8b2zid7a/VOCdevkit2012.tar.gz?dl=0) and extract it into your `$AffordanceNet_Context_ROOT/data` folder; And download this [Annotations](https://www.dropbox.com/home/gt/IVAlab/Deep_Learning_Project/data/affordanceNovel/Annotations_objectness) containing xml with `objectness` instead of all objects to replace `$AffordanceNet_Context_ROOT/data/VOCdevkit2012/VOC2012/Annotations`; And download this file on [dropbox](https://www.dropbox.com/s/zfgn3jo8b2zid7a/VOCdevkit2012.tar.gz?dl=0) and extract it into your `$AffordanceNet_Context_ROOT/data/cache` folder; Make sure you use the category split on [dropbox](https://www.dropbox.com/sh/bahp8aci3ejpytx/AAAlLD1L31XVuOSPzffNJkHya?dl=0) and extract it into your `$AffordanceNet_Context_ROOT/data/VOCdevkit2012/VOC2012/ImageSets/Main` folder
	- You will need the VGG-16 weights pretrained on imagenet. For your convinience, please find it [here](https://www.dropbox.com/s/i4kv0vgn078d1jb/VGG16.v2.caffemodel?dl=0)
	- Put the weight into `$AffordanceNet_Context_ROOT/imagenet_models`
	- If you want novel instance split, please find it [here](https://www.dropbox.com/sh/ya5n61prbc8ftum/AABABu3mqQW438BldvVUYmwoa?dl=0)

2. Train AffordanceNet_Context:
```
cd $AffordanceNet_Context_ROOT
./experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 pascal_voc
```


### Physical Manipulation with affordance
- trained model for DEMO on [dropbox](https://www.dropbox.com/s/2pymk87dzu1io24/vgg16_faster_rcnn_iter.caffemodel?dl=0) 
- put under `./pretrained/`

1.1. Install [Freenect](https://github.com/OpenKinect/libfreenect)


2.1  Run detection
```
cd $AffordanceNet_ROOT/scripts
python demo_img_socket_noprocess_firstAff_kinect.py
```
- You should see the output to be detected point in 3D 
- Specify `affordance_id` for your need.  
```
affordance_id = 6 # 1: grasp 2:cut 3: scoop 4: contain 5:pound 6: support 7:wrap-grasp
```

### Physical Manipulation with PDDL
1.1. Install [Fast-Downward](https://github.com/danfis/fast-downward) for PDDL.

1.2. Install [ROS](http://wiki.ros.org/ROS/Introduction).

1.3. Install [Freenect](https://github.com/OpenKinect/libfreenect)

1.4. Compile [ivaHandy](https://github.com/ivaROS/ivaHandy) in your ros workspace `handy_ws` for our Handy manipulator.

1.5. Compile [handy_experiment](https://github.com/ivaROS/handy_experiment) in your ros workspace `handy_ws` for experiment codebase.

1.6. Train your own object detector (try [tf-faster-rcnn by endernewton](https://github.com/endernewton/tf-faster-rcnn)). [model](https://drive.google.com/file/d/1ji1c554ZFmGMP6028NKTTSjjDOokeAgd/view?usp=sharing), [weights](https://drive.google.com/file/d/1KZ2e56VXnIlbH4YWR_EaH7TFHVA3wJMa/view?usp=sharing)


2.1. run Handy (our robot, you may check our codebase and adjust yours)
```
cd handy_ws
roslaunch handy_experiment pickplace_pddl.launch
```
2.2. run camera
```
roslaunch freenect_launch freenect.launch depth_registration:=true
```

2.3. run PDDL, object detector (keep running it for 2.4 2.5 2.6 2.7)
```
cd $AffordanceNet_ROOT/scripts
python kinect_pddl_UMD_firstAffordance_objectness_contain_objdetection_sub.py
```

2.4. run PDDL, spoon or knife into bowl
```
cd $AffordanceNet_ROOT/scripts
python kinect_pddl_UMD_firstAffordance_objectness_contain_objdetection_spoon_or_knife_in_bowl.py
```

2.5. run PDDL, spoon or trowl scoop coffee
```
cd $AffordanceNet_ROOT/scripts
python kinect_pddl_UMD_firstAffordance_objectness_contain_objdetection_spoon_or_knife_in_bowl.py
```

2.6. run PDDL, spoon to plate to bowl
```
cd $AffordanceNet_ROOT/scripts
python kinect_pddl_UMD_firstAffordance_objectness_contain_objdetection_spoon_to_plate_to_bowl.py
```
and 
```
python kinect_pddl_UMD_firstAffordance_objectness_contain_objdetection_spoon_to_plate_to_bowl2.py
```

2.7. run PDDL, objects into containers
```
cd $AffordanceNet_ROOT/scripts
python kinect_pddl_UMD_firstAffordance_objectness_contain_objdetection_objects_into_containers.py
```
and 
```
python kinect_pddl_UMD_firstAffordance_objectness_contain_objdetection_objects_into_containers2.py
```


Note you might need to:

(1) modify camera parameters:
```
KINECT_FX = 494.042
KINECT_FY = 490.682
KINECT_CX = 330.273
KINECT_CY = 247.443
```
(2) modify the relative translation from aruco tag to robot base:
```
obj_pose_3D.position.x = round(coords_3D[0], 2) + 0.20
obj_pose_3D.position.y = round(coords_3D[1], 2) + 0.30
obj_pose_3D.position.z = round(coords_3D[2], 2) - 0.13 
```

(3) modify a good range for your object scale:
```
(arr_rgb.shape[0] > 100 and arr_rgb.shape[1] > 100)
```

(4) modify the `args.sim` path for debug mode




### License
MIT License

### Acknowledgment
This repo borrows tons of code from
- [affordanceNet](https://github.com/nqanh/affordance-net) by nqanh


### Contact
If you encounter any questions, please contact me at fujenchu[at]gatech[dot]edu


### Modifications
1. [Annotations](https://www.dropbox.com/home/gt/IVAlab/Deep_Learning_Project/data/affordanceNovel/Annotations_objectness) contains xml with `objectness` instead of all objects, (and corresponding model descriptions for two classes)   
2. Modify proposal_target_layer.py
3. to modify affordance number: (1) no prototxt: "mask_score" (2) no config: __C.TRAIN.CLASS_NUM = 13 (3) no proposal_target_layer: label_colors (4) yes proposal_target_layer: label2dist



