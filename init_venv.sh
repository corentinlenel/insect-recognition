#!/bin/bash

######################################
# init_env.sh
# Utility:  set up  a virtual python environment for 
#           tensorflow on linux in the same directory
# Use:      ./init_env.sh -virtual_env_name(optional)
# Auteur:   Corentin LENEL
# Update:   15/04/2020
######################################


# Test parameter presence

if [ -z $1 ]; then
    NAME_VENV="tensorflow"
else
    NAME_VENV="$1"
fi

# Ubuntu/Linux 64-bit

# Installation of the tools

sudo apt-get install python3-pip python3-dev -y
pip3 install --upgrade pip
pip3 install virtualenv

# Test GPU presence

nvidia_presence=$(ldconfig -p | grep -c nvidia)

if [ "$nvidia_presence" -gt 0 ]; then

    # GPU enabled

    echo "Carte graphique NVIDIA présente"
    virtualenv -p python3 "$NAME_VENV"
    source "$NAME_VENV/bin/activate"
    pip3 install tensorflow-gpu
    pip3 install --upgrade tensorflow

else

    # CPU only

    echo "Pas de carte graphique présente"
    virtualenv --system-site-packages "$NAME_VENV"
    source "~/$NAME_VENV/bin/activate"
    pip3 install tensorflow
    pip3 install --upgrade tensorflow


fi

pip install matplotlib
pip install pillow
sudo apt install python3-tk

deactivate

clear

cat << EOF
End of the initialisation of the environment

You can activate your environment with the command :
source $NAME_VENV/bin/activate

Then, you can launch your python script using tensorflow tools

If you want to exit of the environment use the command :
deactivate

EOF
