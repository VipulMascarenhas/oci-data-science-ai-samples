#Deploying onnx and pytorch multi models using Triton[WIP]
##Overview
In this sample, we will deploy 2 models on NVIDIA Triton Inference Server using OCI Data Science Model Deployment. One model is a DenseNet model in an ONNX format, the other model is a ResNet model on PyTorch. The purpose of this sample is to showcase the ability of Triton Inference Server to deploy multiple models on the same server, even when they are using different frameworks.

##Step 1: Set up Triton Inference Server
###Step 1.1: Create Model Artifact
To use Triton, we need to build a model repository. The structure of the repository as follows:

```
models
|
+-- densenet_onnx
|    |
|   +-- config.pbtxt
|   +-- 1
|        |
|        +-- model.onnx
+-- resnet
|.    |
|.   +--config.pbtxt
|    +-- 1
|         |
|         +--model.pt
```
The config.pbtxt configuration file is optional. The configuration file is autogenerated by Triton Inference Server if the user doesn't provide it. If you are new to Triton, it is highly recommended to review Part 1 of the conceptual guide.

```
mkdir -p models/densenet_onnx/
```

```
mkdir -p models/resnet/1
```

####DenseNet Model
```
wget -O model_repository/densenet_onnx/1/model.onnx \

     https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx
 ```

####Resnet Model

#####Creating resnet50 model

```
import torch

# Load the PyTorch model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

# Set the model to evaluation mode
model.eval()
# Define an example input tensor
input_data = torch.randn(1, 3, 224, 224)

# Trace the PyTorch model and generate a TorchScript model
traced_model = torch.jit.trace(model, input_data)
# Save the TorchScript model to a file
traced_model_path = "model.pt"
traced_model.save(traced_model_path)
```


```
cp model.pt models/resnet/1
```



#####Creating config.pbtxt for Resnet model

```
name: "resnet"
platform: "pytorch_libtorch"
max_batch_size: 1

input [
{
name: "input",
data_type: TYPE_FP32,
dims: [ 3, 224, 224 ]
}
]

output [
{
name: "output",
data_type: TYPE_FP32,
dims: [ 1000 ]
}
]

instance_group [
{
kind: KIND_CPU
}
]
```

```
cp config.pbtxt models/resnet
```

###Step 1.2  Upload NVIDIA based triton server image to OCI Container Registry (OCIR)

####Creating image locally
```
mkdir -p tritonServer
cd tritonServer
git clone https://github.com/triton-inference-server/server.git -b v2.30.0 --depth 1
cd server
python compose.py --backend onnxruntime --backend pytorch --repoagent checksum --output-name $(OCIR_REGION).ocir.io/$(OCIR_NAMESPACE)/oci-datascience-triton-server/onnx-pytorch-runtime:1.0.0
```
Refer  https://docs.oracle.com/en-us/iaas/data-science/using/mod-dep-byoc.htm#construct-container for details on uploading image to OCIR



###Step 1.3 Upload model artifact to Model catalog

Compress model_repository folder created in Step 1.1 in zip format and upload it to model catalog via python sdk. Refer https://docs.oracle.com/en-us/iaas/data-science/using/models_saving_catalog.htm for details



###Step 1.4 Create Model Deployment
OCI Data Science Model Deployment supports Triton Inference Server as a special container, mapping the service-mandated endpoints to the Triton's inference and health HTTP/REST endpoint to free you from having to do so. To Enable it, set the following environment variable when creating the Model Deployment:

```
CONTAINER_TYPE = TRITON
```


####Using python sdk

```
# create a model configuration details object
model_config_details = ModelConfigurationDetails(
model_id= <model_id>,
bandwidth_mbps = <bandwidth_mbps>,
instance_configuration = <instance_configuration>,
scaling_policy = <scaling_policy>
)

# create the container environment configuration
environment_config_details = OcirModelDeploymentEnvironmentConfigurationDetails(
environment_configuration_type="OCIR_CONTAINER",
environment_variables={'CONTAINER_TYPE': 'TRITON'},
image="iad.ocir.io/testtenancy/oci-datascience-triton-server/onnx-runtime:1.0.0",
image_digest="sha256:aa32690a166b09015d34c9372812ee9c878cbdc75649f7be6e4465b5eb9ad290",
cmd=[
"/opt/nvidia/nvidia_entrypoint.sh",
"tritonserver",
"--model-repository=/opt/ds/model/deployed_model"
],
server_port=8000,
health_check_port=8000
)

# create a model type deployment
single_model_deployment_config_details = data_science.models.SingleModelDeploymentConfigurationDetails(
deployment_type="SINGLE_MODEL",
model_configuration_details=model_config_details,
environment_configuration_details=environment_config_details
)

# set up parameters required to create a new model deployment.
create_model_deployment_details = CreateModelDeploymentDetails(
display_name= <deployment_name>,
model_deployment_configuration_details = single_model_deployment_config_details,
compartment_id = <compartment_id>,
project_id = <project_id>
)
```


## Step 2: Using python-sdk to query  the Inference Server
Install dependencies & download an example image to test inference.
```
wget  -O ${HOME}/img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```



Firstly, specify the json inference payload with input and output layers for the model as well as describe the shape and datatype of the expected input and output.


###Inference request for densenet onnx model

```
from PIL import Image
import numpy as np
import json
# Load the image and resize it to 224x224
img = Image.open(<path to image>).resize((224, 224))
# Convert the image to a 3D NumPy array
img_array = np.array(img)

# Add a batch dimension to the array
input_data = np.expand_dims(img_array, axis=0)

# Convert the array to the appropriate data type
input_data = input_data.astype(np.float32).toList()


request_body = {"inputs": [{"name": "data_0", "shape": [1,3,224,224], "datatype": "FP32", "data": final_data}], "outputs": [{"name": "fc6_1", "shape":[1,1000],"datatype": "FP32"}]}

request_body = json.dumps(request_body)
```



Secondly, specify the request headers indicating model name and version

```
request_headers = {"model_name":"densenet_onnx", "model_version":"1"}
```


Lastly, we send an inference request to the Triton Inference Server

```
The OCI SDK must be installed for this example to function properly.
Installation instructions can be found here: https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/pythonsdk.htm
import requests
import oci
from oci.signer import Signer

config = oci.config.from_file("~/.oci/config") # replace with the location of your oci config file
auth = Signer(
tenancy=config['tenancy'],
user=config['user'],
fingerprint=config['fingerprint'],
private_key_file_location=config['key_file'],
pass_phrase=config['pass_phrase'])

endpoint = <modelDeploymentEndpoint>

inference_output = requests.request('POST',endpoint, data=request_body, auth=auth, headers=request_headers).json()['outputs'][0]['data'][:5]
```



The output of the same should look like below:

```
[-7.781406879425049, 8.147658348083496, -5.036427021026611, -31.72595977783203, -19.792512893676758]
```



###Inference request for resnet model

```
from PIL import Image

from torchvision import transforms
img = Image.open(<path to image>)
#preprocessing function
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = preprocess(img).numpy().tolist()
body ={"inputs": [{"name": "input", "shape": [1,3,224,224], "datatype": "FP32", "data": img}], "outputs": [{"name": "output", "shape":[1,1000],"datatype": "FP32"}]}

request_body = json.dumps(request_body)
```


Secondly, specify the request headers indicating model name and version

```
request_headers = {"model_name":"resnet", "model_version":"1"}
```



Lastly, we send an inference request to the Triton Inference Server

```
# The OCI SDK must be installed for this example to function properly.
# Installation instructions can be found here: https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/pythonsdk.htm

import requests
import oci
from oci.signer import Signer

config = oci.config.from_file("~/.oci/config") # replace with the location of your oci config file
auth = Signer(
tenancy=config['tenancy'],
user=config['user'],
fingerprint=config['fingerprint'],
private_key_file_location=config['key_file'],
pass_phrase=config['pass_phrase'])

endpoint = <modelDeploymentEndpoint>

inference_output = requests.request('POST',endpoint, data=request_body, auth=auth, headers=request_headers).json()['outputs'][0]['data'][:5]
```


The output of the same should look like below:

```
[-0.26288753747940063, 4.352395534515381, -2.0595359802246094, -2.0003817081451416, -3.181145191192627]
```

##Update Model Deployment
OCI Data Science Model Deployment supports the zero downtime update of individual models without changing the version structure. However,  If user perform update_zdt for triton based model deployments, version structure should be unchanged for underlying model else it will result in downtime.

```
CONTAINER_TYPE = TRITON
```

set the following environment variable when updating  the Model Deployment:

####Using python sdk

```
# create a model configuration details object
model_config_details = ModelConfigurationDetails(
model_id= <model_id>,
bandwidth_mbps = <bandwidth_mbps>,
instance_configuration = <instance_configuration>,
scaling_policy = <scaling_policy>
)

# create the container environment configuration
environment_config_details = OcirModelDeploymentEnvironmentConfigurationDetails(
environment_configuration_type="OCIR_CONTAINER",
environment_variables={'CONTAINER_TYPE': 'TRITON'},
image="iad.ocir.io/testtenancy/oci-datascience-triton-server/onnx-runtime:1.0.0",
image_digest="sha256:aa32690a166b09015d34c9372812ee9c878cbdc75649f7be6e4465b5eb9ad290",
cmd=[
"/opt/nvidia/nvidia_entrypoint.sh",
"tritonserver",
"--model-repository=/opt/ds/model/deployed_model"
],
server_port=8000,
health_check_port=8000
)

# update a model type deployment
single_model_deployment_config_details = data_science.models.SingleModelDeploymentConfigurationDetails(
deployment_type="SINGLE_MODEL",
model_configuration_details=model_config_details,
environment_configuration_details=environment_config_details
)

# set up parameters required to update a model deployment.
update_model_deployment_details = UpdateModelDeploymentDetails(
display_name= <deployment_name>,
model_deployment_configuration_details = single_model_deployment_config_details,
compartment_id = <compartment_id>,
project_id = <project_id>
)
```

##Conclusion
This sample guides you through the deployment of 2 different models from 2 different frameworks onto a Triton Inference Server. You can extend this example to deploy more models from more frameworks into the same Triton Inference Server.
