#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

# 例: src/setup_user.sh に置き換える内容
#!/bin/bash
set -e
source $HOME/miniforge3/etc/profile.d/conda.sh
conda create -n fmri3 python=3.11 -y
conda activate fmri3
pip install --upgrade pip
# 以下は既存の pip install 行を貼る

#pip install --upgrade pip

#python3.11 -m venv fmri3
#source fmri3/bin/activate

pip install numpy matplotlib==3.8.2 jupyter jupyterlab_nvdashboard jupyterlab tqdm scikit-image==0.22.0 accelerate==0.24.1 webdataset==0.2.73 pandas==2.2.0 einops ftfy regex kornia==0.7.1 h5py==3.10.0 open_clip_torch==2.24.0 torchvision==0.16.0 torch==2.1.0 transformers==4.37.2 xformers==0.0.22.post7 torchmetrics==1.3.0.post0 diffusers==0.23.0 deepspeed==0.13.1 wandb omegaconf==2.3.0 pytorch-lightning==2.0.1 sentence-transformers==2.5.1 evaluate==0.4.1 nltk==3.8.1 rouge_score==0.1.2 umap-learn
pip install ipykernel
pip install git+https://github.com/openai/CLIP.git --no-deps
pip install dalle2-pytorch
pip install huggingface-hub==0.34.0
