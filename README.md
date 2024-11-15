# wubai

## run docker image
docker run -it `
    --gpus all `
    -v ${PWD}/data:/data `
    --name magenta-edm-container `
    tensorflow/tensorflow:2.11.0-gpu bash
