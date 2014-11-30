#ifndef _CUDAFUNC_H
#define _CUDAFUNC_H

#include <stdio.h>

// CUDA include files
#include <cuda_runtime_api.h>
#include <cuComplex.h>

#define CUDA_COMPLEX 1
#define CUDA_INPLACE 2
#define CUDA_ZERO_COPY 4
//#define CUDA_DOUBLE  8 
//#define CUDA_INT     16 

// NOTE: I haven't seen a way to query this from the device. Literature suggests
// that this is the max number of blocks per dimension (dim3) on any of the GPUs.
#define CUDA_MAX_BLOCKS 65535

#define CUDA_MAX_STREAMS 10

// Global flag to set whether CUDA has failed at runtime.
static int cuda_runtime_failed=0;

// Global flag to set whether CUDA has failed during the setup process
static int cuda_setup_failed=0;

//{{{ cuda error
// ##################### A common set of error functions for CUDA calls
static void CudaErrorRuntime( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    cuda_runtime_failed=1;
  }
}
#define CUDA_ERROR_RUNTIME( err ) (CudaErrorRuntime( err, __FILE__, __LINE__ ))

static void CudaErrorSetup( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    cuda_setup_failed=1;
  }
}
#define CUDA_ERROR_SETUP( err ) (CudaErrorSetup( err, __FILE__, __LINE__ ))

static void CudaError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
  }
}
#define CUDA_ERROR( err ) (CudaError( err, __FILE__, __LINE__ ))


static void CudaFatal( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    // Exiting in this manner is just rude
    //exit( EXIT_FAILURE );
  }
}
#define CUDA_FATAL( err ) (CudaFatal( err, __FILE__, __LINE__ ))
//}}}

//{{{ standard error

#define INFO(...) (fprintf(stdout,__VA_ARGS__))
#define ERROR(...) (fprintf(stderr,__VA_ARGS__))
#define DEBUG(...) (fprintf(stderr,__VA_ARGS__))
#define WARN(...) (fprintf(stdout,__VA_ARGS__))

//}}}

// The base CUDA plan structure
//{{{ CUDA_PLAN_T
typedef struct {
  cudaDeviceProp prop; // Holds the device properties of the selected CUDA device
  int gpu;             // The device selected by the plan init
  int use_streams;     // Flag whether or not to use streams
  long nelem;          // Number of elements to process with this plan
  int nchunks;         // Number of "chunks" to break the input into
  long elem_per_chunk; // Number of elements to process per "chunk"
  int use_zero_copy;   // Flag whether or not to use zero copy buffers
  int flags;           // A copy of the flags used during initialization
  int cmplx;           // Flag to indicate complex buffers
  int inplace;         // Flag to indicate inplace processing
  int nblocks;         // Number of blocks to break the input into
  int nthreads;        // Number of threads to break the blocks into
  int num_streams;     // Number of streams available on the selected device
  int verbose;         // Verbose flag

  // Set of host buffers
  float *in1;
  float *in2;
  float *out;

  // Set of device buffers
  float *in1_dev[CUDA_MAX_STREAMS];
  float *in2_dev[CUDA_MAX_STREAMS];
  float *out_dev[CUDA_MAX_STREAMS];
} CUDA_PLAN_T;
//}}}

// Function definitions
//{{{ function definitions
CUDA_PLAN_T *cuda_plan_init(long, int, int, int, int, int);
void cuda_plan_destroy(CUDA_PLAN_T *);
void show_devices();
void show_plan(CUDA_PLAN_T *);

// CUDA Vector functions
int cuda_v_cmplx_conj_mult(CUDA_PLAN_T *);
int cuda_v_cmplx_mult(CUDA_PLAN_T *);
int cuda_v_cmplx_div(CUDA_PLAN_T *);
int cuda_v_cmplx_add(CUDA_PLAN_T *);
int cuda_v_cmplx_sub(CUDA_PLAN_T *);
int cuda_v_real_mult(CUDA_PLAN_T *);
int cuda_v_real_div(CUDA_PLAN_T *);
int cuda_v_real_add(CUDA_PLAN_T *);
int cuda_v_real_sub(CUDA_PLAN_T *);


// HOST Vector functions
int host_v_cmplx_conj_mult(CUDA_PLAN_T *);
int host_v_cmplx_mult(CUDA_PLAN_T *);
int host_v_cmplx_div(CUDA_PLAN_T *);
int host_v_cmplx_add(CUDA_PLAN_T *);
int host_v_cmplx_sub(CUDA_PLAN_T *);
int host_v_real_mult(CUDA_PLAN_T *);
int host_v_real_div(CUDA_PLAN_T *);
int host_v_real_add(CUDA_PLAN_T *);
int host_v_real_sub(CUDA_PLAN_T *);


inline long minl(long x, long y){
  return(x < y ? x:y);
}

//}}}

#endif
