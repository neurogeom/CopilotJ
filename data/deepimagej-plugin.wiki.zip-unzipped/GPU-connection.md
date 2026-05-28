# Requirements for the connection with a GPU

The Java libraries used by DeepImageJ can connect with an installed GPU using [CUDA]. Therefore, the connection can be made as long as the required version of CUDA and CUDnn drivers is installed.

The version is determined by the Java libraries used to load the models: Deep Java Library (PyTorch) and TensorFlow Java API.

| Operating system | Supports TensorFlow | Supports PyTorch | Supports GPU connection | GPU computing capability |
| ---------------- | ------------------- | ---------------- | ----------------------- | ------------------------ |
| Windows          | Yes                 | Yes              | Yes                     | 3.5                      |
| Linux            | Yes                 | Yes              | Yes                     | 6                        |
| iOS (Mac)        | Yes                 | Yes              | No\*                    |                          |

\*The TensorFlow java library for GPU is not supported in iOS (Mac) operating systems.

| Deep Learning library   | CUDA versions      |
| ----------------------- | ------------------ |
| TensorFlow 1.15 (GPU)   | 10.0               |
| TensorFlow 1.14 (GPU)   | 10.0               |
| TensorFlow 1.13.1 (GPU) | 10.0               |
| TensorFlow 1.12 (GPU)   | 9.0                |
| PyTorch until 1.10.0    | 10.2 / 11.3        |
| PyTorch until 1.9.1     | 10.2 / 11.1        |
| PyTorch until 1.9.0     | 10.2 / 11.1        |
| PyTorch until 1.8.1     | 10.2 / 11.1        |
| PyTorch until 1.7.1     | 10.1 / 10.2 / 11.0 |
| PyTorch until 1.7.0     | 10.1 / 10.2 / 11.0 |
| PyTorch until 1.6.0     | 10.1 / 10.2        |
| PyTorch until 1.5.0     | 9.2 / 10.1 / 10.2  |
| PyTorch until 1.4.0     | 9.2 / 10.1         |

After selecting the CUDA version, a compatible CuDNN with CUDA has to be installed too. Further information about CuDNN version compatibility according to the NVIDIA hardware and CUDA versions: https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html#abstract

## GPU connection for PyTorch models.

Having other CUDA versions installed might be a source of conflict. In Windows, if a non-compatible version is installed, the plugin will fail to load the model. [This is a known bug](https://github.com/awslabs/djl/discussions/126).

If you experience this error and DeepImageJ can not execute the PyTorch model:

- Remove `CUDA_PATH` (if exists) from your system environment variables.
- Make sure that your PATH does not include any directory or file with the words `Nvidia` or `CUDA`.
  - Go to `Edit the system environment variables` or `Edit environment variables for your account`.
  - Click on `Environment variables`.
  - Check the `Path` and `CUDA_PATH` variables (note that Windows is not case sensitive so they might be written as `PATH` or `path`).

Deep Java Library already supports GPU and CPU computation, so there is no need to update the Java dependency.

## GPU connection for TensorFlow models.

- TensorFlow 1.12: CUDA 9.0 and corresponding CUDnn and drivers.
- TensorFlow 1.15: CUDA 10.0 and corresponding CUDnn and drivers.

Further information about [Tensorflow-CUDA and Pytorch-CUDA correspondences](https://www.tensorflow.org/install/source#tested_build_configurations).

### Update TensorFlow version to support GPU processing

In ImageJ1, the installation of Tensorflow-GPU versions has to be done manually:

- Delete the `libtensorflow-X.X.X.jar` and `libtensorflow_jni-X.X.X.jar` (where `X.X.X` is the version number) from the `jars` folder inside ImageJ directory. **Make sure that there are no other TensorFlow dependencies in ImageJ folders.**
- Go to https://mvnrepository.com/artifact/org.tensorflow/libtensorflow and select the Tensorflow version that you require.
- Go to https://mvnrepository.com/artifact/org.tensorflow/libtensorflow_jni_gpu and select the .jar executable from the same version selected in the previous step.
- Copy both .jar executables into the same folder where the previous .jars were located (`jars`)

In ImageJ2 or Fiji, the version update can be done automatically with the TensorFlow version manager. In ImageJ2/Fiji installation:

- Go to `Edit>Options>Tensorflow` and select the required Tensorflow-GPU version.

  ![](https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/tf-manager.png)

- Restart ImageJ/Fiji and the installation of the selected GPU version will be finished.

Further information can be found at:

- https://github.com/CSBDeep/CSBDeep_website/wiki/CSBDeep-in-Fiji-%E2%80%93-Installation
- https://github.com/tensorflow/tensorflow/issues/16660

# CUDA installation

## CUDA and GPU requirements

- For Windows systems, the GPUs require a minimum compute capability of 3.5.
- In Linux systems, the Tensorflow Java library requires a GPU with compute capability of 6.0, which is relatively big and a known issue (https://github.com/tensorflow/tensorflow/issues/36228).

You can check your GPU compute capability [here](https://developer.nvidia.com/cuda-gpus.)

## Windows

We provide some guidelines to install a specific version of CUDA in your Windows OS. You can also follow the [official documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

### CUDA

- Your computer has to be CUDA capable. Check that your machine fulfills all the requirements to run CUDA with GPU:
  - System requirements: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements.
  - GPU requirements: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#verify-you-have-cuda-enabled-system.

- (Optional) If you can, delete all previous installations of CUDA. Even though CUDA allows having multiple versions of CUDA at the same time, it is advisable to have only one version of CUDA installed to avoid conflicts. As mentioned previously, it is possible and easy to have several CUDA versions installed in the same machine.
- Go to the Cuda Toolkit Archive (https://developer.nvidia.com/cuda-toolkit-archive ) and select the required CUDA distribution, e.g., CUDA 10.0

<img src="https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/cuda-toolkit.png" align="center" width="600"/>

- Select the options that fit your machine. We selected the installer shown in the figure below. You can also choose between a network installer or a local installer.
- Download and execute the installer. Follow the instructions and install all the recommended settings.
- Once the installation is finished, check the environment variables. There should be a new environment variable named ‘CUDA_PATH’ that points to the folder where CUDA was installed, usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0`. In the environment variable `path` there should have been added two new directories, one pointing towards `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin` and the other one towards `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp`.

### CUDnn

To have CUDA accelerated Deep Neural Networks, the installation of another package, CUDnn, is required:

- Go to https://developer.nvidia.com/rdp/cudnn-download or https://developer.nvidia.com/rdp/cudnn-archive (you will have to create an account if you do not already have one).
- Download the corresponding CUDnn version for Windows depending on the CUDA version that you installed following the table above; e.g., for CUDA version 10.0 you would need CUDnn 7.4.1.
- Unzip the downloaded folder. Inside you will find three folders: `lib`, `bin`, and `include`. Move the contents from `lib` to the folder C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib, include to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include and `bin` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin`.

If you want to install another CUDA version, you just need to follow the same steps. To move from one CUDA version to another one, you only need to change the environment variables. Set the environment variable `CUDA_PATH` to the folder containing the desired CUDA version and change the directories of the `PATH` to the corresponding CUDA version.

Note that after installing CUDA version `X.y`, the environment variable `CUDA_PATHXy` will be created. It will always point to the directory containing `CUDA X.y`. You do not need to change it.

## Linux/Unix

We provide some guidelines to install a specific version of CUDA in your Unix/Linux OS. You can also follow the [official documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation).

### CUDA

- Your computer has to be CUDA capable. Check that your machine fulfills all the requirements to run CUDA with GPU:
  - System requirements: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements
  - GPU requirements: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#verify-you-have-cuda-enabled-system.
- Make sure `gcc` is installed in your machine. Check it on the terminal with :
  ```
  gcc --version
  ```
- If an error message appears, install `gcc` on your system. On Ubuntu 20.4:
  ```
  sudo apt update
  sudo apt install build-essential
  ```
- (Optional) If you can, delete all previous installations of CUDA. Even though CUDA allows having multiple versions of CUDA at the same time, it is advisable to have only one version of CUDA installed to avoid conflicts. As mentioned previously, it is possible and easy to have several CUDA versions installed in the same machine.

- Go to the [Cuda Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and select the required CUDA distribution, e.g., CUDA 10.0.
- Select the options that fit your machine. We selected the installer shown in the figure below. You can also choose between several installers.
- Follow the installation instructions provided by Nvidia. In step 2 of the example shown below, in order to know which string you should substitute `<version>` with the directory that appears at `/var/` and look for the directory. I our case (shown in the image) we have two CUDA versions installed.
  - If we were installing version 10.0 the command would be:
    ```
    sudo apt-key add /var/cuda-repo-10.0.130-410.45/7fa2af80.pub
    ```
  - And If we were installing CUDA 9.0
    `     sudo apt-key add /var/cuda-repo-9.0-local/7fa2af80.pub
    `
    <img src="https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/cuda-toolkit2.png" align="center" width="750"/>
    <img src="https://github.com/deepimagej/deepimagej-plugin/raw/master/wiki/test-dij.png" align="center" width="600"/>

- Installation with the `runfile` is similar to the one on Windows. Just accept all the recommended settings.

Once the installation is finished, the `CUDA-X.y/bin` directory must be added to the `PATH`, where `X.y` is the corresponding CUDA version.

To add the variable:

```
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
```

We also need to add another directory to the `LD_LIBRARY_PATH`:

- For 64-bit operating systems:
  ```
  export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64\
  {LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  ```
- For 32-bit operating systems:
  ```
  export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib\
  {LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  ```
  Note that these variables will only stay during the session. Once the computer is restarted, it will vanish. To add them permanently follow the instructions [here](https://askubuntu.com/questions/58814/how-do-i-add-environment-variables).

### CUDnn

To have CUDA accelerated Deep Neural Networks, the installation of another package, CUDnn, is required:

- Go to https://developer.nvidia.com/rdp/cudnn-download or https://developer.nvidia.com/rdp/cudnn-archive (you will have to create an account if you do not already have one) and download the corresponding CUDnn version for Linux depending on the CUDA version that you installed following the table above. For example, for CUDA version 10.0 you would need CUDnn 7.4.1.
- Unzip the downloaded folder.
- Copy the contents into its corresponding folder inside the CUDA directory. Assuming that CUDA was located in the default directory `/usr/local/cuda-10.0`, the commands to execute are:
  ```
  sudo cp -P Downloads/cuda/include/cudnn.h /usr/local/cuda-10.0/include/
  sudo cp -P Downloads/cuda/lib/libcudnn* /usr/local/cuda-10.0/lib64/
  sudo chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*
  ```
- Run the following command to help DeepImageJ finding the newly installed CUDA version:
  ```
  sudo updated
  ```
  With these steps, you can install as many CUDA versions as needed.

# DeepImageJ GPU management (experimental)

Tensoflow Java does not provide a way to know if the model is being used by a GPU. The most common way used to know if your program is being used by the GPU is calling the command

```
nvidia-smi
```

which informs about the works loaded in a GPU and the current usage of the GPU.

To ease the looking-out of GPU usage, DeepImageJ aims to inform the user whether it is connected to a GPU or not.
It performs several checks for the GPU usage:

- Check if there are environment variables in the system that corresponds to a CUDA installation.
- Check that the CUDA version found is compatible with the Tensorflow or Pytorch version being used.
- Call internally the command `nvidia-smi` immediately before and after loading the model. A new process should appear in the console output if the plugin is using the GPU. The console output should also inform that the new process corresponds to an instance of ImageJ.

On the other hand, for DJL Pytorch it is possible to know whether the model is loaded on the GPU or not, thus the information given by the plugin on this issue will be more reliable.

## Limitations

- Due to a [known bug](https://github.com/tensorflow/tensorflow/issues/37289 and https://github.com/tensorflow/tensorflow/issues/19731), Tensorflow does not free the GPU memory when a model is closed, but when the whole process is terminated. The process will not disappear from the GPU until ImageJ1/ImageJ2/Fiji is closed. In order to avoid possible fails derived from this fact, DeepImageJ also checks how many ImageJ processes are open. Thus if `nvidia-smi` shows that the GPU is being used by a process related to ImageJ before and after loading the model (the `nvidia-smi` output does not change), but there is only one ImageJ1/ImageJ2/Fiji process running on the computer, the plugin will understand that it is indeed using the GPU.

Furthermore, loading a process in the GPU does not mean that the model inference is going to be executed on the GPU. There are at least two cases for this:

- If the CUDnn libraries have not been correctly installed or there is a missing environment variable, the model will be loaded and executed in the GPU but its execution will not be enhanced.
- If the compute capability is lower than required, the process will be shown in the `nvidia-smi` command as loaded but it will fall back to CPU. The latter can be checked by running `nvidia-smi` during model inference and checking that the GPU is using memory.

Nevertheless, the most accurate information will be always given by the `nvidia-smi` command.
To avoid misunderstandings, it is always advisable to execute the command at least once each time CUDA or Tensorflow versions are upgraded.

[CUDA]: https://blogs.nvidia.com/blog/2012/09/10/what-is-cuda-2/
