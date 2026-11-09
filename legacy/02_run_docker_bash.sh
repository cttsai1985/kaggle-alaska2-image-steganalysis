DOCKER_IMAGE=cttsai1985/ml-env-torch-vision

CPU_COUNT=8
GPU_DEVICE='"device=0"'

SHM_SIZE=2G

RootSrcPath=${PWD}
DockerRootSrcPath=/root/src/

DataPath=${PWD}/input
DockerDataPath=/root/src/input

DataCachedPath=${PWD}/input/models-alaska2-image-steganalysis
DockerDataCachedPath=/root/src/input/models-alaska2-image-steganalysis

OutputPath=${PWD}/output
DockerOutputPath=/root/src/output

RootPort1=8008
DockerRootPort1=8888

RootPort2=6006
DockerRootPort2=6666

WORKDIR="/root/src/script"

docker rm $(docker ps -a -q)

CMD="jupyter notebook --port ${DockerRootPort1} --ip=0.0.0.0 --allow-root --no-browser"
CMD="bash"

echo docker -i -t --cpus=$CPU_COUNT --gpus ${GPU_DEVICE} -p $RootPort1:$DockerRootPort1 -p $RootPort2:$DockerRootPort2 -e PYTHONPATH=/root/src -v ${RootSrcPath}:${DockerRootSrcPath} -v $(readlink -f ${DataPath}):${DockerDataPath} -v $(readlink -f ${DataCachedPath}):${DockerDataCachedPath} -v $(readlink -f ${OutputPath}):${DockerOutputPath} --shm-size $SHM_SIZE --workdir=${WORKDIR} $DOCKER_IMAGE $CMD

docker run -i -t --cpus=$CPU_COUNT --gpus ${GPU_DEVICE} -p $RootPort1:$DockerRootPort1 -p $RootPort2:$DockerRootPort2 -e PYTHONPATH=/root/src -v ${RootSrcPath}:${DockerRootSrcPath} -v $(readlink -f ${DataPath}):${DockerDataPath} -v $(readlink -f ${DataCachedPath}):${DockerDataCachedPath} -v $(readlink -f ${OutputPath}):${DockerOutputPath} --shm-size $SHM_SIZE --workdir=${WORKDIR} $DOCKER_IMAGE $CMD
