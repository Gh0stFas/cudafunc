# CUDAFUNC #

A set of library routines that provide stream-lined access to CUDA enabled
GPU's.

## Purpose ##

While the CUDA interface itself is very easy to pick up and master,
I found myself writing the same bits of code each and every time I wanted
to access a CUDA enabled GPU for vector operations. This library brings
together, in an FFTW-like format, a host of common vectorized operations.
This library is intended for individuals who may not want to dive into the
details of a GPU, and just want to get started utilizing one for fast vector
operations....or those individuals such as myself who are familiar with GPU's
and don't want to have to dive into the details every time we put something
together.

## Installing/Building ##

Pull the latest source from GitHub

```sh
git clone git://github.com/Gh0stFas/cudafunc.git
```

The current installation assumes that the CUDA SDK is already installed
so all you need to do is simply run

```sh
make
```

This will build the library along with a simple test binary which you can
run to test the library out along with utilize as an example of usage.
