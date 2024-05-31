# vit-aws

## Outline - 
- Loading ViT.
- Loading training data and after Preprocessing pushed to AWS S# bucket.
- Training ViT on AWS SageMaker.
- Torch JIT (Just In Time) Scripting, Quantizing and converting the model to Pytorch Lite Format which helps in lesser memory requirements hence efficient memory usage,fast inference time hence lower latency.
    - These optimizations are done so that the model can perform well if decided to deployed to production environment.
- Deploying the model with AWS SageMaker endpoint for public use.


## Further Improvements - 
- Loading the PyTorch model using  ONNX format ,i.e the C++ format for more faster inference.
- Extend the deployed SageMaker endpoint with Flask APIs.
- More rigorous evaluation metrics.