#source /work/gaohy/mlir.sh

if [ -d build ]; then
	rm -rf build
fi

cmake -B build && cd build && make -j
