NV_GPU=${1:-3} nvidia-docker run -it --rm -p ${2:-6888}:${3:-7888} -v $PWD:/hear_and_see -v ${4:-/data/Projects/hear_and_see/data}:/hear_and_see/data socraticdatum/hear_and_see:dev
