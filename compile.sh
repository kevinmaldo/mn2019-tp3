git submodule init
git submodule update
mkdir -p build || true
rm -rf build/*
cd build
cmake -DPYTHON_EXECUTABLE="$(which python3)" -DCMAKE_BUILD_TYPE=Release ..
make install
