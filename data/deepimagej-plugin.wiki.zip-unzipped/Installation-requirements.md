## Automatic installation with Fiji update sites (optional for ImageJ2 and Fiji distributions)

In Fiji or Image2:

- In Fiji/ImageJ, click on `Help > Update...`.
- Once the ImageJ Updater pops up, click on `Manage update sites > Add update site`.
- There you can write the name ("DeepImageJ") and the URL given (https://sites.imagej.net/DeepImageJ/).
- Click on `Close`.
- Click on `Apply Changes`.
  Then The plugin will then be installed automatically together with its java dependencies
  **WARNING!:** If you are using Windows, the previous steps will only let you using Tensorflow models. For PyTorch models, you still need to follow one more step: [Visual studio installation](https://github.com/deepimagej/deepimagej-plugin/wiki/Installation-requirements#pytorch-dependencies-for-windows-os).

## Manual installation (mandatory when using ImageJ1)

- Download the latest release of DeepImageJ at [GitHub releases](https://github.com/deepimagej/deepimagej-plugin/releases). The ZIP file (`DeepImageJ.zip`) contains all the necessary libraries (JAR files) and DeepImageJ_X.X.X.jar.
- Unzip the ZIP file.
- Copy the plugin executable `DeepImage_X.X.X.jar` in the `plugins` folder of ImageJ1 / Image2 / Fiji directory.
- The folder `Dependencies` contains all the Java dependencies (.jar) needed. You can take all the dependencies and drag&drop them directly in ImageJ. Whenever it asks where to store them:
  - If you are in ImageJ1, in `ImageJ/plugins/jars`.
  - Otherwise, in the `jars` folder inside ImageJ/Fiji.
- Otherwise, you can directly copy&paste all the dependencies in the locations specified above.

The last step might produce some version conflicts with existing libraries in your local installation. Thus being careful is advised. If there are already other versions of the dependencies inside the `jars` folder, some conflicts might appear when ImageJ2/Fiji is started, and the plugin might not function correctly.

# Dependencies of DeepImageJ

The `.jar` executables included in the Dependencies folder and needed to run the plugin are:

- api-0.9.0.jar
- pytorch-native-auto-1.7.0.jar
- pytorch-engine-0.9.0.jar
- imagej-tensorflow-1.1.6.jar
- kotlin-stdlib-1.3.72.jar
- libtensorflow-1.15.0.jar
- libtensorflow_jni-1.15.0.jar (libtensorflow_jni_gpu-1.15.0.jar for GPU support)
- npy-0.3.3.jar
- proto-1.15.0.jar
- protobuf-java-3.5.1.jar
- snake-yaml-1.2.1.jar

Note that for ImageJ1 we do not need the `ImageJ-Tensorflow-1.6.0 jar` as it corresponds to the TensorFlow version manager that allows changing Tensorflow versions from ImageJ2/Fiji. This feature is available for the ImageJ2/Fiji distributions, but not for the ImageJ1.

## PyTorch dependencies for Windows OS

Pytorch is available for MacOs and Linux/Unix Operating Systems after the common installation. However, to use PyTorch models in Windows OS it is necessary to install Visual Studio 2019 redistributable:

- Go to [https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)
- Download Visual Studio 2019 redistributables and install them:
  - Go to [https://visualstudio.microsoft.com/es/downloads/](https://visualstudio.microsoft.com/es/downloads/) and install Visual Studio 2019

    ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/visual-studio.png)

## Update Tensorflow version

The [ImageJ-TensorFlow](https://imagej.net/libs/tensorflow) manager is available in Fiji or ImageJ2 distributions. It allows the automatic upgrade of TensorFlow versions. See how to use the TensorFlow manager in Fiji [here](https://github.com/deepimagej/deepimagej-plugin/wiki/GPU-connection#update-tensorflow-version-to-support-gpu-processing).

The Tensorflow version can be manually updated as well:

### Update TensorFlow version manually (mandatory for ImageJ1)

In order to switch between Tensorflow versions in ImageJ, the `libtensorflow-1.15.0.jar` and `libtensorflow_jni-1.15.0.jar` (or `libtensorflow_jni_gpu-1.15.0.jar`) dependencies need to be replaced by the executable .jar file corresponding to the desired version.
All the available versions can be found at

- [https://mvnrepository.com/artifact/org.tensorflow/libtensorflow](https://mvnrepository.com/artifact/org.tensorflow/libtensorflow)
- [https://mvnrepository.com/artifact/org.tensorflow/libtensorflow_jni](https://mvnrepository.com/artifact/org.tensorflow/libtensorflow_jni) - Or [https://mvnrepository.com/artifact/org.tensorflow/libtensorflow_jni_gpu](https://mvnrepository.com/artifact/org.tensorflow/libtensorflow_jni_gpu) for GPU support.

**Note that for the moment DeepImageJ only supports Tensorflow until version `1.15.0`.**

## Update PyTorch version manually

There is no PyTorch manager in ImageJ yet, so PyTorch version updates need to be done manually. DeepImageJ uses the [Deep Java Library](https://docs.djl.ai/) (DJL) to allow deploying Pytorch models in Java. DJL supports Pytorch versions from 1.4 to 1.8. Unfortunately, in order to switch Pytorch versions, DJL executables need to be changed. They can be downloaded from [here](https://mvnrepository.com/artifact/ai.djl.pytorch). At the moment DeepImageJ comes by default with Pytorch 1.7.

To change the Pytorch version, three .jar files need to be replaced:

- pytorch-native-auto-x.y.z.jar
- pytorch-engine-X.Y.Z.jar
- api-X.Y.Z.jar

where x.y.z refers to the Pytorch version and X.Y.Z refers to the DJL version.

Below the table of compatibilities between versions can be found:

| Pytorch version                                 | Pytorch engine version    | DJL API version |
| ----------------------------------------------- | ------------------------- | --------------- |
| pytorch-native-auto-1.4.0.jar                   | pytorch-engine-0.4.0.jar  | api-0.4.0.jar   |
| pytorch-native-auto-1.4.0.jar                   | pytorch-engine-0.5.0.jar  | api-0.5.0.jar   |
| pytorch-native-auto-1.5.0.jar                   | pytorch-engine-0.6.0.jar  | api-0.6.0.jar   |
| pytorch-native-auto-1.6.0.jar                   | pytorch-engine-0.7.0.jar  | api-0.7.0.jar   |
| pytorch-native-auto-1.6.0.jar                   | pytorch-engine-0.8.0.jar  | api-0.8.0.jar   |
| pytorch-native-auto-1.7.0.jar                   | pytorch-engine-0.9.0.jar  | api-0.9.0.jar   |
| pytorch-native-auto-1.7.1.jar                   | pytorch-engine-0.11.0.jar | api-0.11.0.jar  |
| pytorch-native-auto-1.8.0.jar                   | pytorch-engine-0.11.0.jar | api-0.11.0.jar  |
| pytorch-native-auto-1.9.0.jar                   | pytorch-engine-0.13.0.jar | api-0.14.0.jar  |
| pytorch-native-auto-(1.8.1, 1.9.0, 1.9.1).jar   | pytorch-engine-0.14.0.jar | api-0.14.1.jar  |
| pytorch-native-auto-(1.8.1, 1.9.1, 1.10.0).jar  | pytorch-engine-0.15.0.jar | api-0.15.0.jar  |
| pytorch-native-auto-(1.8.1, 1.9.1, 1.10.0).jar  | pytorch-engine-0.16.0.jar | api-0.16.0.jar  |
| pytorch-native-auto-(1.9.1, 1.10.0, 1.11.0).jar | pytorch-engine-0.17.0.jar | api-0.17.0.jar  |
| pytorch-native-auto-(1.9.1, 1.10.0, 1.11.0).jar | pytorch-engine-0.18.0.jar | api-0.18.0.jar  |
