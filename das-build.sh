# add code to loop download submodule
i=0
while ((i<10))
do
    echo "this is the $i times download"
    git submodule update --init --recursive
    if [ "$?" == 0 ];then
	echo "download submodule success"
        break
    fi
    ((i++))
done
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
pip3 install -r requirements.txt
pip3 install -r requirements-dev.txt
USE_HCU=ON USE_CUDA=OFF python3 -m build --wheel --no-isolation --config-setting=cmake.build-type=Release --config-setting=install.strip=true
