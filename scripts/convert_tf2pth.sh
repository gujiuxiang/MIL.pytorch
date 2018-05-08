#! /bin/sh
# pip install mmdnn
convert_tf_2_pytorch()
{
    if [ ! -d tmp ]; then
        mkdir tmp
    fi
    # Extract tf model files
    python -m mmdnn.conversion.examples.tensorflow.extract_model -n resnet_v1_101 -ckpt /home/jxgu/github/MIL.pytorch/model/oidv2_resnet_v1_101/oidv2-resnet_v1_101.ckpt
    # Convert tf to IR
    python -m mmdnn.conversion._script.convertToIR -f tensorflow -d kit_imagenet -n imagenet_resnet_v1_101.ckpt.meta --dstNodeName Squeeze -w imagenet_resnet_v1_101.ckpt
    # Convert IR to Pytorch
    python -m mmdnn.conversion._script.IRToCode -f pytorch --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy -dw kit_pytorch.npy
    # Dump the PyTorch model
    python -m mmdnn.conversion.examples.pytorch.imagenet_test --dump resnet.pth -n kit_imagenet.py -w tmp/kit_pytorch.npy
}

convert_tf_2_pytorch