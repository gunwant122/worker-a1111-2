
# ---------------------------------------------------------------------------- #
#                         Stage 1: Download the models                         #
# ---------------------------------------------------------------------------- #
FROM alpine/git:2.43.0 as download

# NOTE: CivitAI usually requires an API token, so you need to add it in the header
#       of the wget command if you're using a model from CivitAI.
RUN apk add --no-cache wget && \
    wget -q -O /model.safetensors 'https://huggingface.co/SEVUNX/sdxl_model/resolve/9ef3d61c0768695513b4b8c53003d347c93c71af/download_2/protovisionXLHighFidelity3D_releaseV660Bakedvae.safetensors'

# ---------------------------------------------------------------------------- #
#                        Stage 2: Build the final image                        #
# ---------------------------------------------------------------------------- #
FROM python:3.10.14-slim as build_final_image

ARG A1111_RELEASE=v1.9.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN export COMMANDLINE_ARGS="--skip-torch-cuda-test --precision full --no-half"
RUN export TORCH_COMMAND='pip install ---no-cache-dir torch==2.1.2+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'

RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget libgoogle-perftools-dev libtcmalloc-minimal4 procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

RUN --mount=type=cache,target=/cache --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    ${TORCH_COMMAND} && \
    pip install --no-cache-dir xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

COPY --from=download /model.safetensors /model.safetensors

# Install RunPod SDK
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir runpod

ADD src .
COPY builder/cache.py /stable-diffusion-webui/cache.py
RUN cd /stable-diffusion-webui && python cache.py --use-cpu=all --ckpt /model.safetensors


# install controlnet from git clone --depth=1 https://github.com/Mikubill/sd-webui-controlnet.git extensions/sd-webui-controlnet
RUN git clone --depth=1 https://github.com/Mikubill/sd-webui-controlnet.git /stable-diffusion-webui/extensions/sd-webui-controlnet
# install requirements.txt
RUN cd /stable-diffusion-webui/extensions/sd-webui-controlnet && pip install -r requirements.txt
# save the model diffusers_xl_canny_full.safetensors https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_full.safetensors
#  create the folder models/ControlNet if it doesn't exist 
# https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors
RUN mkdir -p /stable-diffusion-webui/models/ControlNet
RUN wget -q -O stable-diffusion-webui/models/ControlNet/diffusers_xl_canny_full.safetensors https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_full.safetensors
RUN wget -q -O stable-diffusion-webui/models/ControlNet/diffusion_pytorch_model.safetensors  https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors
# Set permissions and specify the command to run
RUN chmod +x /start.sh
CMD /start.sh 
