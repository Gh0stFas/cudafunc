CC = nvcc
INC = ./inc
LIB = ./lib
BIN = ./bin
OPT = -O2
LIBFLAGS = ${OPT} -c -I${INC} -D_GNU_SOURCE
# Starting with GCC 4.6, I've had to place the -Wl,--no-as-needed option for
# the linker so it can link the math library properly. Without this option you
# would have to put the math library in an order dependent manner....bleh
CFLAGS = ${opt} -lm -L${LIB} -I${INC} -D_GNU_SOURCE
OBJ = libcudafunc.a
EXECUTABLE = cudatest

all : ${EXECUTABLE}

${EXECUTABLE} : ${OBJ} cudatest.cu ${INC}/cudafunc.h ${LIB}/cudafunc.cu
	${CC} -o cudatest cudatest.cu ${CFLAGS} -lcudafunc

libcudafunc.a : ${LIB}/cudafunc.cu ${INC}/cudafunc.h
	${CC} ${LIBFLAGS} -o ${LIB}/cudafunc.o ${LIB}/cudafunc.cu
	ar -cvq ${LIB}/libcudafunc.a ${LIB}/cudafunc.o
	
clean :
	/bin/rm -f cudatest ${LIB}/*.a ${LIB}/*/*.o ${BIN}/*

install :
	/bin/mkdir -p ${BIN}
	/bin/mv cudatest ${BIN}
