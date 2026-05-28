# Pyramidal pooling models in DeepImageJ

DeepImageJ supports image processing networks with miscellaneous inputs and outputs. Some region proposal networks (RPN) or pyramidal feature pooling like [RetinaNet][retinanet] are interesting use-cases for detection, instance segmentation, or panoptic segmentation tasks. In particular, the [Mask R-CNN][mask_rcnn] is a well-known architecture that has proven a great potential to perform panoptic segmentation. The inference with these networks is slightly different from other architectures as they have different inputs that determine what is the output deployed by the model.

- The Mask R-CNN extracts many features of the input image and generates candidate bounding boxes of different sizes and aspect ratios. These bounding boxes are scaled to a specific size so they can be processed. Then, regression and classification of the bounding boxes are performed, so it is decided whether it contains an object of interest and which object-class it represents. Finally, the winning bounding boxes are processed to extract the binary masks of the object contained on each of the bounding boxes.

To generate the proper inputs, DeepImageJ uses pre- and post-processing routines in Java. Take a look at [our GitHub repository][mrcnn_DIJ] with the code for the [Mask R-CNN implemented in Keras](https://github.com/matterport/Mask_RCNN).

## Build a DeepImageJ bundled model of the Mask R-CNN

In the following lines, we detail the steps to build a DeepImageJ bundled model of a trained Mask R-CNN.

- Drag and drop the folder containing the trained model.
- Click on the checkbox that asks if the model corresponds to a pyramidal pooling model

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-1.png)

- You will see a list of all the inputs and outputs defined in the [Mask R-CNN Keras model][keras].
- **Inputs**
  - Input_image: the image tensor that is processed. For the moment DeepImageJ only supports 2D images for these kind of networks.
  - Input_image_meta: vector tensor where each of the entries corresponds to a label for each of the classes. Labels go from `0` to `n – 1`, where `n` is the number of the different classes used in the classification.
  - Input_anchors: matrix containing the proposed vertices for the regions (bounding boxes) where features are searched. The vertices are proposed at each of the pooling layers.

- **Outputs needed for inference**
  - Mrcnn_detection: matrix tensor containing the location of the object detected and the probability/score of its classification. The matrix has 6 columns and `n` rows, where `n` is the maximum number of objects that can be detected by the model. If the model only detects `m` objects in an image, `n-m` rows are going to be 0`s. Each row contains information for one object detected. The first four rows are the vertices of the bounding boxes of the objects detected. The coordinates are normalized and given in pixel units. The 5th row indicates the label of the object detected, i.e. the class. The last row corresponds to the probability of the object belonging to the class in which it was classified.
  - Mrcnn_mask: image output of the model. It is a stack of images of a fixed size. In the case of [matterport](https://github.com/matterport/Mask_RCNN) it is 28x28 with as many channels as classes, and as many slices as objects can be detected. Thus for a model that detects 81 classes and up to 100 objects per image, the stack will have 100 slices of 28x28 images of 81 channels. This mask is used to reconstruct the mask of the original input image using the information provided by the `mrcnn_detection` output. Each row of `mrcnn_detection` contains information for the corresponding slice. And the label number indicates the channel that contains the correct segmentation of that object.
    - For example, for row 5, which has classified an object as class number 1, the pixels at `slice = 5`, `channel = 1` will be the scaled mask for an object in the image. This mask can be de-escaled from 28x28 pixels into its real size using the bounding box coordinates provided in the first four columns of the row.

- **Outputs needed only for training (not in DeepImageJ)**
  - Rpn_class: matrix used during training. It is optimized during training but it is not necessary during inference.
  - ROI: matrix used during training. It is optimized during training but it is not necessary during inference.
  - Mrcnn_box: matrix used during training. It is optimized during training but it is not necessary during inference.
  - Rpn_box: matrix used during training. It is optimized during training but it is not necessary during inference.

This is how the correct specification of inputs and outputs for the Mask R-CNN Keras model would look like:

![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-2.png)
![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-3.png)

- In the next step, the size of the image **after pre-processing** has to be introduced. The Mask R-CNN usually reshapes the input image to a predetermined size, performs the needed operations and finally the post-processing resizes the result into the original image dimensions. Thus, the size of the image after pre-processing has to be introduced in the field _Minimum Size_. On the other hand, _Step Size_ has to be 0 everywhere as the network only allows a specific size.

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-4.png)

- The plugin then asks for specfications in the raw outputs of the model. For _lists_ outputs, the dimensions corresponding to rows and columns need to be selected (although this should be already done in the _Tensor Organization_ step).

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-5.png)

- Then, the fixed size of the raw ouput image, before post-processing, has to be specified. With respect to the dimensions, or axes, **N/i/z** refers to the maximum objects that can be detected by the network, **Y and X** to the size of the scaled bounding boxes that may contain objects and **C** is the number of classes the network is trained to identify.

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-6.png)

- The following step asks for the pre-processing files. The DeepImageJ team implemented the pre- and post-processing routines for the [Mask R-CNN in Java][mrcnn_DIJ] mimicking the [Mask R-CNN in Keras implemented in this Github repository][keras]. We also provide the .jar executable file in any of the ready-to-use Mask R-CNN models ([Mask R-CNN for RGB street images][objects] or [Usiigaci-Mask R-CNN][usiigaci]).
  The Java pre-processing (either the one provided by DeepImageJ or any other) has to be provided in the pre-processing step. A Macro pre-processing file can also be specified. Regard that if there are two files, the one on top is excuted first.

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-7.png)

- After clicking on _Next_, a dialog box will appear asking for external files needed for the Java processing. In the case of DeepImageJ's Mask R-CNN, a configuration file must be provided. [Here][config] is an example of the config file needed. Using it with a different Mask R-CNN is just an issue of changing the corresponding parameters according to the model training.

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-8.png)

- The post-processing step is very similar to the previous one. The post-processing Java executable should be provided, along with a Macro post-processing file (if there is any). Regard that DeepImageJ's Mask R-CNN .jar file contains both the pre- and post-procesing, so the same file should be added again, now in the post-processing step.

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-9.png)

- Then, after clicking on _Next_, a dialog box will appear asking for external files or Java executables. Analogously to the pre-processing step, the configuration file with the correct parameters has to be provided. Again, for the DeepImageJ Mask R-CNN, the config file is the same for both pre- and post-processing, so it should be added for a second time here.

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-10.png)

**After all these steps, your Mask R-CNN should be ready to go!**

## Build a DeepImageJ bundled model of the Mask R-CNN

The team of DeepImageJ implemented in Java the pre- and post-processing routines for the Mask R-CNN in a general way, so they could be re-used by model developers without coding. Model developers only need to worry about adapting the [configuration file][config] with the parameters used in their training.

The pre- and post-processing where translated to Java from Python using [Matterpot's Mask R-CNN][keras] as a reference, with the same parameters as in their [configuration file][keras_config]. Thus, if the pre- and post-processing of your Mask R-CNN differ greatly from [Matterpot's Mask R-CNN][keras], it might not be directly compatible with DeepImage's Mask R-CNN. However, a glance at the [code][mrcnn_DIJ] might help adjusting the processing to your model.

In the case that the model is compatible, the only issue you should take care of is the configuration file. Our config file contains all the parameters needed to define the Mask R-CNN and adapting them to your model should be enough to run your model on DeepImageJ.
Here is the example of the config files of two different Mask R-CNNs, [one detects cells][usiigaci] and can only identify between cell and background and the other [detects common life objects][objects] and can identify 81 different classes. Both models use the same Java processing and the only difference between models is the configuration file, as it is shown below:

![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-11.png)

[mask_rcnn]: https://ieeexplore.ieee.org/document/8237584
[retinanet]: https://ieeexplore.ieee.org/document/8237586
[keras]: https://github.com/matterport/Mask_RCNN
[mrcnn_DIJ]: https://github.com/deepimagej/deepimagej-java-processing
[objects]: https://zenodo.org/record/4155785/files/MaskRCNN.bioimage.io.model.zip?download=1
[usiigaci]: https://bioimage.io/#/?partner=deepimagej&type=all&id=deepimagej%2Fusiigaci
[config]: https://github.com/deepimagej/deepimagej-java-processing/blob/main/Mask%20R-CNN/config.ijm
[keras_config]: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
