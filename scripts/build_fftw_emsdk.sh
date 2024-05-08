
FFTW_ARCHIVE_FILE=fftw-3.3.10.tar.gz
FFTW_ARCHIVE_DIR=fftw-3.3.10
if [ ! -f "$FFTW_ARCHIVE_FILE" ]; then
    echo "Downloading FFTW"
    wget "https://www.fftw.org/fftw-3.3.10.tar.gz"
fi
if [ ! -d "$FFTW_ARCHIVE_DIR" ]; then
    echo "Unpacking FFTW archive"
    tar -xf "$FFTW_ARCHIVE_FILE"
fi
cd "$FFTW_ARCHIVE_DIR"
emconfigure ./configure --disable-fortran --prefix=/mnt/c/Users/gfo/Personnel/choreo/build/test/ CFLAGS="-s WASM=1" CXXFLAGS="-s WASM=1"  LDFLAGS="-s WASM=1"  #--with-combined-threads
emmake make -j
emmake make install

