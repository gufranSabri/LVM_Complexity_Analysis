#!/bin/bash

# Run ViT model
python src/main.py --model vit  --batch_size 128
if [ $? -ne 0 ]; then
  echo "Error running ViT model. Exiting."
  exit 1
fi

# Run Swin model
python src/main.py --model swin --batch_size 128
if [ $? -ne 0 ]; then
  echo "Error running Swin model. Exiting."
  exit 1
fi

# Run ResNet model
python src/main.py --model resnet --batch_size 128
if [ $? -ne 0 ]; then
  echo "Error running ResNet model. Exiting."
  exit 1
fi

echo "All models ran successfully!"
