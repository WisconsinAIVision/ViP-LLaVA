git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

# Set AZCOPY concurrency to auto
echo "export AZCOPY_CONCURRENCY_VALUE=AUTO" >> ~/.zshrc
echo "export AZCOPY_CONCURRENCY_VALUE=AUTO" >> ~/.bashrc

# Activate conda by default
echo ". /home/vscode/miniconda3/bin/activate" >> ~/.zshrc
echo ". /home/vscode/miniconda3/bin/activate" >> ~/.bashrc

# Use vip-llava environment by default
echo "conda activate vip-llava" >> ~/.zshrc
echo "conda activate vip-llava" >> ~/.bashrc

# Add dotnet to PATH
echo 'export PATH="$PATH:$HOME/.dotnet"' >> ~/.bashrc
echo 'export PATH="$PATH:$HOME/.dotnet"' >> ~/.zshrc

# Create and activate vip-llava environment
source /home/vscode/miniconda3/bin/activate
conda create -y -q -n vip-llava python=3.10
conda activate vip-llava

# Install Nvidia Cuda Compiler
conda install -y -c nvidia cuda-compiler

pip install pre-commit==3.0.2

# Install package locally
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# Install additional packages for training
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

# Download checkpoints to location outside of the repo
git clone https://huggingface.co/mucai/vip-llava-7b ~/vip-llava-7b

# Commented because it is unlikely for users to have enough local GPU memory to load the model
# git clone https://huggingface.co/mucai/vip-llava-13b ~/vip-llava-13b

echo "postCreateCommand.sh COMPLETE!"
