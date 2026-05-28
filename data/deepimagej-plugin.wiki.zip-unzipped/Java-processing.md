# Java pre- and post-processing

DeepImageJ is compatible with pre- and post-processings in Java. Java routines can be faster and more efficient than simple ImageJ macro scripts. However, packing these routines together with a Deep Learning model in a ready-to-use manner is slightly more difficult.

DeepImageJ loads Java external programs in a similar way to what ImageJ does with external plugins. To create a Java pre- and post-processing routines follow the next steps:

- Create a new Java project. You can create a standard Java program or use build tools such as Gradle or Maven.

- Add the `DeepImageJ-2.1.X.jar`([here](https://github.com/deepimagej/deepimagej-plugin/releases/download/2.1.4-SNAPSHOT/DeepImageJ_-2.1.4-SNAPSHOT.jar)) and `ij.jar` ([here](https://github.com/imagej/imagej1)) dependencies using the most suitable version for your scripts. You will need to use methods and objects from those executables. Note that the accessibility to Java code is a new feature in the DeepImageJ 2.1 release, i.e. previous versions of DeepImageJ will not work with this. In the caption below, those jars are added to the path, as it is a simple Java project:

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/java-1.png)

- The program has to contain a class that implements the DeepImageJ interface `deepimagej.processing.PreProcessingInterface` (for pre-processing routines) or `deepimagej.processing.PostProcessingInterface` (for post-processing routines). DeepImageJ will then recognize the methods `deepImageJPreprocessing ` or `deepImageJPostprocessing` as the methods to apply to the images in the pre- or post-processing, respectively.

- Java pre- can be used to generate inputs for the model. The method `deepImageJPreprocessing` has to return a `HashMap<String, Object>` where each of the keys is named after each of the inputs to the model. Depending on the definition of the model input given at the _Tensor Organization Step_, the object in the pre-processing output HashMap can be a class or another.

- For `Images`, the pre-processing routine should give an **ImageJ ImagePlus** and for `Parameters`, either **Tensorflow Tensors**, **Deep Java Library NDArrays** or **ImageJ ResultsTables**.
  Thus, the ImagageJ .jar file should always be in the classpath, and if the processing uses tensors, you should add the `libtensorflow-X.Y.Z.jar` and `libtensorflow_jni-X.Y.Z.jar` (where `X.Y.Z` is the version for your model) to the build path. If it includes NDArrays, the Deep Java Library dependencies should be added instead. The dependencies be managed as you want, either in the traditional way of building the path or using more advanced dependency managers as Maven.

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/java-2.png)

- Similarly to pre-processing, the post-processing method takes as input another `HashMap<String, Object>`, where each key is named after each of the outputs of the model. The objects can be either an ImagePlus, if the output was defined as `Image` in DeepImageJ Build Bundled Model, or they can be an ImageJ `ResultsTable` if the output was defined as a `List`. Then, in the post-processing program, you can use those objects to perform any action. Note that ResultsTable only has rows and columns, thus the number of output dimensions for a tensor is limited to 2. Only 3 dimensions are allowed if one of them represents the batch. Nonetheless, the batch size should always be 1.
  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/java-3.png)

- Furthermore, if for complex processing routines, the program might need the help of external Java libraries. In that case, it is necessary to include the path to the library in the dialog box that appears right after the pre- and post- processing step in the Build Bundled Model plugin. DeepImageJ will not pack the libraries into the model to avoid licensing issues but when the model is deployed with DeepImageJ Run, it will go through the ImageJ/Fiji classpath looking for the corresponding library to check if it is present or not. If it is not it will show a error message asking the user to download the corresponding library and to locate it on the jars folder.

- Here is an example of a pre-processing using external libraries and what should be specified in Build Bundled Model. Regard that we do not need to include either ImageJ or Tensorflow libraries (or any of the required libraries by DeepImageJ), despite being needed by the program. Those libraries should be already in the Fiji/ImageJ distribution in order to DeepImageJ to work.

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/java-4.png)
  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/java-5.png)

- DeepImageJ also supports the use of external files in the Java pre- or post-processing. Those files have to be specified in the dialog box that opens after the pre- and post-processing steps in the Build Bundled Model plugin. The files are then managed in the method `setConfigFiles` both at `deepimagej.processing.PreProcessingInterface` and `deepimagej.processing.PostProcessingInterface`. These files, unlike Java executables, will be copied into the model folder because they are necessary for the model execution.

- In the figure below we can see a configuration file being provided for pre-processing.

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/dij-maskrcnn-10.png)

- The 'setConfigFiles' method is fed an array of paths to each of the auxiliary files provided. This array excludes .jar and .class files as they are considerered dependencies of the pre- or post-processing. Also, regard that two different files with the same name cannot be provided. The model will package both files in the model folder and one will overwrite the other.

- In the example below we can observe how the interface method 'setConfigFiles' retrieves the file of interest by its name and then proceeds to read it and parse it, setting the parameters as variables of the class that can be read by every method of the class.

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/java-6.png)

- Once the script is finished make an executable. It can be either a `.jar` or a `.class` file.

The DeepImageJ team provides an example of complete Java pre- and post-processing for the Mask R-CNN [here](https://github.com/deepimagej/deepimagej-java-processing/tree/main/Mask%20R-CNN). The documentation on how to pack a model with a complex Java processing such as the Mask R-CNN is also [provided](https://github.com/deepimagej/deepimagej-plugin/wiki/Region-proposal-and-pyramidal-feature-pooling-networks).

- **Warning:** Debugging inside the DeepImageJ is not possible, so make sure the program works before using it directly within DeepImageJ.
