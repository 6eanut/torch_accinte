```
pip install torch==2.9.1
# cuda 12.2 needs gcc-12, g++-12

git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout 2.9.1

cd test/cpp_extensions/open_registration_extension
git clone https://github.com/6eanut/torch_accinte.git
cd torch_accinte
export CPATH=pathto/pytorch:pathto/pytorch/test/cpp_extensions/open_registration_extension/third_party/openreg:$CPATH
python -m pip install --no-build-isolation .
```
