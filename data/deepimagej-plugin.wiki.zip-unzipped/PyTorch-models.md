# PyTorch models in deepImageJ

The easiest way to get a deepImageJ-compatible PyTorch model is by using the [bioimage.io core library](https://github.com/bioimage-io/core-bioimage-io-python) from the [BioImage Model Zoo](https://bioimage.io/#/). Please, follow the [example notebook](https://github.com/bioimage-io/core-bioimage-io-python/blob/main/example/model_creation.ipynb) to see how to do it.

More technically, deepImageJ can load Pytorch models by making use of a third-party library called [Java Deep Learning Library (JDLL)](https://arxiv.org/abs/2306.04796). To load PyTorch models in JDLL, **the models need to be saved in TorchScript format** so the library can make use of the [Python C++ API](https://pytorch.org/cppdocs/). The latter does not add complexity to coding in Python as [it only implies adding 2 extra lines of code](https://djl.ai/docs/pytorch/how_to_convert_your_model_to_torchscript.html):

```python
import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18(pretrained=True)

# Switch the model to eval model
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Save the TorchScript model
traced_script_module.save("traced_resnet_model.pt")
```

**The compatibility of deepImageJ with PyTorch versions is defined by the JDLL**.

For compatibility with Windows OS, [JDLL requires the installation of Visual Studio 2019 redistributable](https://github.com/deepimagej/deepimagej-plugin/wiki/Installation-requirements#pytorch-dependencies-for-windows-os).
