# TensorFlow models in DeepImageJ

## Export a TensorFlow (or Keras) model

DeepImageJ can load TensorFlow models thanks to the TensorFlow Java API developed by the TensorFlow team:

- TensorFlow Java: https://www.tensorflow.org/install/lang_java_legacy
- GitHub repository for TensorFlow models >= 2.3.1: https://github.com/tensorflow/java

As deepImageJ is synchronized with ImageJ's TensorFlow manager, it can only load TensorFlow models until version 1.15. Exceptionally, some TensorFlow models trained with versions <= 2.2 can be loaded.

TensorFlow models have to be stored as [`SavedModel`](https://www.tensorflow.org/tutorials/keras/save_and_load#savedmodel_format) (also called `SavedModelBundle`). When doing so, the saved model is self-contained, i.e. we do not need the original Python code to build the model architecture. To load this model deepImageJ, it is important to use always the same tags, therefore, in TensorFlow <= 1.15, we can use the next code lines to store the model:

```python
from tensorflow import saved_model
from keras.backend import get_session

builder = saved_model.builder.SavedModelBuilder('saved_model/my_model')
signature = saved_model.signature_def_utils.predict_signature_def(
             # dictionary of 'name' and model inputs (it can have more than one)
            inputs={'input': model.input},
            # dictionary of 'name' and model outputs (it can have more than one)
            outputs={'output': model.output})
signature_def_map = {saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
builder.add_meta_graph_and_variables(get_session(), [saved_model.tag_constants.SERVING], signature_def_map=signature_def_map)
builder.save()
```

Note that the TensorFlow Java API calls to the same con in C++ as the TensorFlow library in Python, so the performance of the models in deepImageJ is ensured.

##Load a TensorFlow model in Python saved as `saved_model` bundled model

The information you need to load a tensorflow model in Tensorflow 1.15 or earlier versions:

- Path to the model that contains a `saved_model.pb` file and the `variables` folder.
- Name of the input tensor.
- Name of the output tensor.
- Tag used to store the model (usually `tf.saved_model.tag_constants.SERVING`)

```python
import tensorflow as tf
import numpy as np

input_key = 'input' # name of the input restored from the model.yaml
output_key = 'output' # name of the input restored from the model.yaml
export_path = "/content/tensorflow_saved_model_bundle" # path to the folder containing the model

tag = tf.saved_model.tag_constants.SERVING # it's usually this one but otherwise it's given in config/deepimagej field in model.yaml.
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY # it's usually this one but otherwise it's given in config/deepimagej field in model.yaml.

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        # load the model
        meta_graph_def = tf.saved_model.loader.load(
            sess,
            [tag],
            export_path,
        )
        # Get the input-output dictionary
        signature = meta_graph_def.signature_def
        # Get model input and output tensor names in the graph
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name
        # Get restored model input and output
        input_tensor = graph.get_tensor_by_name(x_tensor_name) # name of the input tensor in the model.yaml
        output_tensor = graph.get_tensor_by_name(y_tensor_name) # name of the input tensor in the model.yaml

        # Random input dataset
        input_data = np.random.rand(1, 256, 256, 8, 1)
        input_data[:,150:200, 150:200] = 1

        # Run prediction
        output_array = sess.run(output_tensor, {input_tensor:input_data})
```
