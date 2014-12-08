#include <cudafunc.h>

//{{{ print_device_prop
void print_device_prop(cudaDeviceProp prop){
  INFO("Name: \t\t\t%s\n",prop.name);
  INFO("Compute Capability: \t%d.%d\n",prop.major,prop.minor);
  INFO("Integrated: \t\t%s\n",(prop.integrated == 1 ? "\033[92mYes\033[0m":"\033[91mNo\033[0m"));
  INFO("Can map host mem: \t%s\n",(prop.canMapHostMemory == 1 ? "\033[92mYes\033[0m":"\033[91mNo\033[0m"));
  INFO("Clock rate: \t\t%d\n",prop.clockRate);
  INFO("Memory clock rate: \t%d\n",prop.memoryClockRate);
  INFO("Concurrent Kernels: \t%s\n",(prop.concurrentKernels == 1 ? "\033[92mYes\033[0m":"\033[91mNo\033[0m"));
  INFO("MP Count: \t\t%d\n",prop.multiProcessorCount);
  INFO("Max threads per MP: \t%d\n",prop.maxThreadsPerMultiProcessor);
  INFO("Max threads per block: \t%d\n",prop.maxThreadsPerBlock);
  INFO("Total const mem: \t%d\n",prop.totalConstMem);
  INFO("Total global mem: \t%d\n",prop.totalGlobalMem);
  INFO("Warp size: \t\t%d\n",prop.warpSize);
  INFO("Async engine count: \t%d\n",prop.asyncEngineCount);
  INFO("Max grid size x: \t%d\n",prop.maxGridSize[0]);
  INFO("Max grid size y: \t%d\n",prop.maxGridSize[1]);
  INFO("Max grid size z: \t%d\n",prop.maxGridSize[2]);
  INFO("Max threads x: \t\t%d\n",prop.maxThreadsDim[0]);
  INFO("Max threads y: \t\t%d\n",prop.maxThreadsDim[1]);
  INFO("Max threads z: \t\t%d\n",prop.maxThreadsDim[2]);
}
//}}}

//{{{ show_devices
void show_devices(){
  cudaDeviceProp prop;

  int count;
  cudaError cuda_error;

  cuda_error = cudaGetDeviceCount(&count);

  if(cuda_error == cudaSuccess){
    INFO("There are %d devices available\n",count);
    for(int i=0;i<count;i++){
      cuda_error = cudaGetDeviceProperties(&prop,i);
      INFO("------------ Device %d ------------\n",i);
      print_device_prop(prop);
    }
  } else ERROR("An error occurred while attempting to retrieve available devices: %s\n",cudaGetErrorString(cuda_error));
}
//}}}

//{{{ cuda_free_buffers
void cuda_free_buffers(CUDA_PLAN_T *p){
  if(p->in1 != NULL) {
    CUDA_ERROR(cudaFreeHost(p->in1));
    p->in1=NULL;
  }
  if(p->in2 != NULL) {
    CUDA_ERROR(cudaFreeHost(p->in2));
    p->in2=NULL;
  }
  if(p->out != NULL) {
    CUDA_ERROR(cudaFreeHost(p->out));
    p->out=NULL;
  }
  if(!p->use_zero_copy){
    int i;
    for(i=0;i<p->num_streams;i++){
      CUDA_ERROR(cudaFree(p->in1_dev[i]));
      CUDA_ERROR(cudaFree(p->in2_dev[i]));
      CUDA_ERROR(cudaFree(p->out_dev[i]));
      p->in1_dev[i]=NULL;
      p->in2_dev[i]=NULL;
      p->out_dev[i]=NULL;
    }
  }
}
//}}}

//{{{ show_plan
void show_plan(CUDA_PLAN_T *p){
  if(p->verbose > 0){
    INFO("******************* CUDA Plan *******************\n");
    INFO("Complex:\t\t%s\n",(p->cmplx == 1 ? "\033[92mYes\033[0m":"\033[91mNo\033[0m"));
    INFO("Zero-copy:\t\t%s\n",(p->use_zero_copy == 1 ? "\033[92mYes\033[0m":"\033[91mNo\033[0m"));
    INFO("In-place:\t\t%s\n",(p->inplace == 1 ? "\033[92mYes\033[0m":"\033[91mNo\033[0m"));
    INFO("Streams Enabled:\t%s\n",(p->use_streams == 1 ? "\033[92mYes\033[0m":"\033[91mNo\033[0m"));
    INFO("Nstreams:\t\t%d\n",p->num_streams);
    INFO("Nchunks:\t\t%d\n",p->nchunks);
    INFO("Elem. per chunk:\t%ld\n",p->elem_per_chunk);
    INFO("Elem. left-over:\t%d\n",p->elem_leftover);
    INFO("Nthreads:\t\t%d\n",p->nthreads);
    INFO("Nblocks:\t\t%d\n",p->nblocks);
    INFO("*************************************************\n");
  }
}
//}}}

//{{{ cuda_plan_init
CUDA_PLAN_T * cuda_plan_init(long nelem, int dev_num, int nblocks, int nthreads, int flags, int verbose){
  CUDA_PLAN_T *p;
  int i;

  if((p = (CUDA_PLAN_T *) calloc(1,sizeof(CUDA_PLAN_T))) == NULL){
    ERROR("Failed to allocate memory for the CUDA plan\n");
  }
  else {
    // Initialize the plan
    // TODO: Need to check this to make sure it's > 0
    p->nelem=nelem;
    p->in1=NULL;
    p->in2=NULL;
    p->out=NULL;
    for(i=0;i<CUDA_MAX_STREAMS;i++){
      p->in1_dev[i]=NULL;
      p->in2_dev[i]=NULL;
      p->out_dev[i]=NULL;
    }
    p->flags=flags;
    p->nblocks=nblocks;
    p->nthreads=nthreads;
    p->verbose=verbose;
    p->elem_leftover=0;

    // Parse the flags
    p->cmplx=(flags&0x1);
    p->inplace=(flags&0x2)>>1;
    p->use_zero_copy=(flags&0x4)>>2;

    memset(&(p->prop),0,sizeof(cudaDeviceProp));

    if(dev_num > -1){
      // If a specific device was requested then assume the user knows
      // what they're doing and blindly set the device and grab the
      // properties.
      p->gpu = dev_num;
      CUDA_ERROR_SETUP(cudaSetDevice(p->gpu));
      CUDA_ERROR_SETUP(cudaGetDeviceProperties(&(p->prop),p->gpu));
      if(p->verbose > 0){
        INFO("------------ Selected Device ------------\n");
        print_device_prop(p->prop);
      }
    }
    else {
      // Check for a device that can meet the libraries maximum requirements
      p->prop.major=3;
      p->prop.minor=0;

      CUDA_ERROR_SETUP(cudaChooseDevice(&(p->gpu),&(p->prop)));
      CUDA_ERROR_SETUP(cudaSetDevice(p->gpu));
      CUDA_ERROR_SETUP(cudaGetDeviceProperties(&(p->prop),p->gpu));
      if(p->verbose > 0){
        INFO("------------ Selected Device ------------\n");
        print_device_prop(p->prop);
      }
    }

    // If this is the case then you will realize no gain in using multiple streams
    if(!(p->prop.deviceOverlap)) {
      WARN("Device does not support overlap. Use of streams has been turned off.\n");
      p->use_streams=0;
    }
    else p->use_streams=1;

    // Number of available streams
    // If we aren't using streams then set it to one
    // TODO: Work out the stream processing path to use more than 2 streams
    p->num_streams=(p->use_streams ? min(p->prop.multiProcessorCount,2):1);

    // Allocate the memory for the buffers
    // NOTE: This process will determine how the follow-on calculations
    // will run. The idea here is that we are going to attempt to do
    // everything using the memory on the GPU itself. In the case that
    // the allocation fails then we will fall back to using zero-copy
    // memory buffers which are simply host memory blocks which are pinned
    // and thus allow access via a hardware device such as the GPU. It
    // should be mentioned that in tests it appears that zero-copy
    // buffers are more efficient then copying blocks of memory on and
    // off the card. This is something to consider in the future. This
    // should be exposed as an option where the user can explicitely
    // request zero-copy buffers.
    int nfloats=(p->cmplx ? 2:1);

    if(!p->use_zero_copy){
      // We're going to set our number of chunks to be a multiple of our number of streams
      p->nchunks = p->num_streams;
      // NOTE: This needs to be set for the nthreads and nblocks calculation which follows
      p->elem_per_chunk=nelem/p->nchunks;
      p->elem_leftover = (int)roundf((((float)nelem/(float)p->nchunks) - (float)p->elem_per_chunk)*(float)p->nchunks);

      // NOTE: In order to use streams the memory must be pinned, or page-locked, due to the use
      // of cudaMemcpyAsync. This is why cudaHostAlloc is used instead of a simple malloc or calloc.
      CUDA_ERROR_SETUP(cudaHostAlloc((void **)&(p->in1),p->nelem*sizeof(float)*nfloats,cudaHostAllocDefault));
      CUDA_ERROR_SETUP(cudaHostAlloc((void **)&(p->in2),p->nelem*sizeof(float)*nfloats,cudaHostAllocDefault));
      if(!p->inplace) CUDA_ERROR_SETUP(cudaHostAlloc((void **)&(p->out),p->nelem*sizeof(float)*nfloats,cudaHostAllocDefault));

      for(i=0;i<p->num_streams;i++){
        // Allocate the memory on the GPU for each stream.
        CUDA_ERROR_SETUP(cudaMalloc((void **) &(p->in1_dev[i]), p->elem_per_chunk*sizeof(float)*nfloats));
        CUDA_ERROR_SETUP(cudaMalloc((void **) &(p->in2_dev[i]), p->elem_per_chunk*sizeof(float)*nfloats));
        if(!p->inplace) CUDA_ERROR_SETUP(cudaMalloc((void **) &(p->out_dev[i]), p->elem_per_chunk*sizeof(float)*nfloats));
      }
      
      // If the setup failed then go ahead and get rid of any allocations the succeeded
      if(cuda_setup_failed) cuda_free_buffers(p);
    }

    if(cuda_setup_failed || p->use_zero_copy){
      // Reset the flag
      if(cuda_setup_failed){
        cuda_setup_failed=0;
        p->use_zero_copy=1;
        WARN("Failed to allocate buffers on the GPU. Falling back to zero-copy host buffers.\n");
      }
      else if(p->verbose > 0) INFO("Setting up user requested zero-copy host buffers.\n");

      // We're not using streams with zero copy buffers so we only have 1 "chunk" to process
      p->nchunks = 1;
      // NOTE: This needs to be set for the nthreads and nblocks calculation which follows
      p->elem_per_chunk=nelem/p->nchunks;

      // Can we even use zero-copy?
      if(p->prop.canMapHostMemory){
        // NOTE: In order for zero-copy buffers work the memory must be pinned, or page-locked by cudaHostAlloc.
        CUDA_ERROR_SETUP(cudaHostAlloc((void **) &(p->in1), p->nelem*sizeof(float)*nfloats,cudaHostAllocWriteCombined|cudaHostAllocMapped));
        CUDA_ERROR_SETUP(cudaHostAlloc((void **) &(p->in2), p->nelem*sizeof(float)*nfloats,cudaHostAllocWriteCombined|cudaHostAllocMapped));
        if(!p->inplace) CUDA_ERROR_SETUP(cudaHostAlloc((void **) &(p->out), p->nelem*sizeof(float)*nfloats,cudaHostAllocWriteCombined|cudaHostAllocMapped));

        // Get pointers to these buffers which work with a GPU
        CUDA_ERROR_SETUP(cudaHostGetDevicePointer(&(p->in1_dev[0]), p->in1, 0));
        CUDA_ERROR_SETUP(cudaHostGetDevicePointer(&(p->in2_dev[0]), p->in2, 0));
        if(!p->inplace) CUDA_ERROR_SETUP(cudaHostGetDevicePointer(&(p->out_dev[0]), p->out, 0));
      } else {
        // TODO: In this case we ought to setup a generic method which
        // will work on any GPU configuration.
        ERROR("Device can not map host memory.\n");
        cuda_setup_failed=1;
      }
    }

    // Auto-calc the number of blocks and the number of threads per block
    // if not explicitely requested
    // NOTE: Here we will always attempt to maximize the number of threads
    // per block.
    if(p->nthreads <= 0) p->nthreads=minl(p->elem_per_chunk,p->prop.maxThreadsPerBlock);
    if(p->nblocks <= 0) p->nblocks = p->elem_per_chunk/p->nthreads;

  }

  // If the CUDA setup failed then go ahead and free the plan if it
  // was created
  if(cuda_setup_failed){
    cuda_plan_destroy(p);
    return NULL;
  } else return(p);
}
//}}}

//{{{ cuda_plan_destroy
void cuda_plan_destroy(CUDA_PLAN_T *p){
  if(p != NULL){
    cuda_free_buffers(p);
    free(p);
    p=NULL;
  }

  return;
}
//}}}

//{{{ HOST
//{{{ complex
//{{{ kernels
void cmplx_conj_mult_kernel_host(cuComplex *a, cuComplex *b, cuComplex *c, long N){
  for(int i=0;i<N;i++) c[i] = cuCmulf(cuConjf(a[i]),b[i]);
}

void cmplx_conj_mult_kernel_host_ip(cuComplex *a, cuComplex *b, long N){
  for(int i=0;i<N;i++) b[i] = cuCmulf(cuConjf(a[i]),b[i]);
}

void cmplx_mult_kernel_host(cuComplex *a, cuComplex *b, cuComplex *c, long N){
  for(int i=0;i<N;i++) c[i] = cuCmulf(a[i],b[i]);
}

void cmplx_mult_kernel_host_ip(cuComplex *a, cuComplex *b, long N){
  for(int i=0;i<N;i++) b[i] = cuCmulf(a[i],b[i]);
}

void cmplx_div_kernel_host(cuComplex *a, cuComplex *b, cuComplex *c, long N){
  for(int i=0;i<N;i++) c[i] = cuCdivf(a[i],b[i]);
}

void cmplx_div_kernel_host_ip(cuComplex *a, cuComplex *b, long N){
  for(int i=0;i<N;i++) b[i] = cuCdivf(a[i],b[i]);
}

void cmplx_add_kernel_host(cuComplex *a, cuComplex *b, cuComplex *c, long N){
  for(int i=0;i<N;i++) c[i] = cuCaddf(a[i],b[i]);
}

void cmplx_add_kernel_host_ip(cuComplex *a, cuComplex *b, long N){
  for(int i=0;i<N;i++) b[i] = cuCaddf(a[i],b[i]);
}

void cmplx_sub_kernel_host(cuComplex *a, cuComplex *b, cuComplex *c, long N){
  for(int i=0;i<N;i++) c[i] = cuCsubf(a[i],b[i]);
}

void cmplx_sub_kernel_host_ip(cuComplex *a, cuComplex *b, long N){
  for(int i=0;i<N;i++) b[i] = cuCsubf(a[i],b[i]);
}

//}}}

//{{{ host_v_cmplx_conj_mult 
int host_v_cmplx_conj_mult(CUDA_PLAN_T *p){
  int status=0;

  if(p != NULL){
    if(p->inplace){
      // Run the cudaKernel
      cmplx_conj_mult_kernel_host_ip((cuComplex *)(p->in1),
                                     (cuComplex *)(p->in2),
                                     p->nelem);

    }
    else{
      // Run the cudaKernel
      cmplx_conj_mult_kernel_host((cuComplex *)(p->in1),
                                  (cuComplex *)(p->in2),
                                  (cuComplex *)(p->out),
                                  p->nelem);
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ host_v_cmplx_mult 
int host_v_cmplx_mult(CUDA_PLAN_T *p){
  int status=0;

  if(p != NULL){
    if(p->inplace){
      // Run the cudaKernel
      cmplx_mult_kernel_host_ip((cuComplex *)(p->in1),
                                     (cuComplex *)(p->in2),
                                     p->nelem);

    }
    else{
      // Run the cudaKernel
      cmplx_mult_kernel_host((cuComplex *)(p->in1),
                                  (cuComplex *)(p->in2),
                                  (cuComplex *)(p->out),
                                  p->nelem);
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ host_v_cmplx_div
int host_v_cmplx_div(CUDA_PLAN_T *p){
  int status=0;

  if(p != NULL){
    if(p->inplace){
      // Run the cudaKernel
      cmplx_div_kernel_host_ip((cuComplex *)(p->in1),
                                     (cuComplex *)(p->in2),
                                     p->nelem);

    }
    else{
      // Run the cudaKernel
      cmplx_div_kernel_host((cuComplex *)(p->in1),
                                  (cuComplex *)(p->in2),
                                  (cuComplex *)(p->out),
                                  p->nelem);
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ host_v_cmplx_add
int host_v_cmplx_add(CUDA_PLAN_T *p){
  int status=0;

  if(p != NULL){
    if(p->inplace){
      // Run the cudaKernel
      cmplx_add_kernel_host_ip((cuComplex *)(p->in1),
                                     (cuComplex *)(p->in2),
                                     p->nelem);

    }
    else{
      // Run the cudaKernel
      cmplx_add_kernel_host((cuComplex *)(p->in1),
                                  (cuComplex *)(p->in2),
                                  (cuComplex *)(p->out),
                                  p->nelem);
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ host_v_cmplx_sub
int host_v_cmplx_sub(CUDA_PLAN_T *p){
  int status=0;

  if(p != NULL){
    if(p->inplace){
      // Run the cudaKernel
      cmplx_sub_kernel_host_ip((cuComplex *)(p->in1),
                                     (cuComplex *)(p->in2),
                                     p->nelem);

    }
    else{
      // Run the cudaKernel
      cmplx_sub_kernel_host((cuComplex *)(p->in1),
                                  (cuComplex *)(p->in2),
                                  (cuComplex *)(p->out),
                                  p->nelem);
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}
//}}}

//{{{ real
//{{{ kernels
void real_mult_kernel_host(float *a, float *b, float *c, long N){
  for(int i=0;i<N;i++) c[i] = a[i]*b[i];
}

void real_mult_kernel_host_ip(float *a, float *b, long N){
  for(int i=0;i<N;i++) b[i] = a[i]*b[i];
}

void real_div_kernel_host(float *a, float *b, float *c, long N){
  for(int i=0;i<N;i++) c[i] = a[i]/b[i];
}

void real_div_kernel_host_ip(float *a, float *b, long N){
  for(int i=0;i<N;i++) b[i] = a[i]/b[i];
}

void real_add_kernel_host(float *a, float *b, float *c, long N){
  for(int i=0;i<N;i++) c[i] = a[i]+b[i];
}

void real_add_kernel_host_ip(float *a, float *b, long N){
  for(int i=0;i<N;i++) b[i] = a[i]+b[i];
}

void real_sub_kernel_host(float *a, float *b, float *c, long N){
  for(int i=0;i<N;i++) c[i] = a[i]-b[i];
}

void real_sub_kernel_host_ip(float *a, float *b, long N){
  for(int i=0;i<N;i++) b[i] = a[i]-b[i];
}

//}}}

//{{{ host_v_real_mult 
int host_v_real_mult(CUDA_PLAN_T *p){
  int status=0;

  if(p != NULL){
    if(p->inplace){
      // Run the cudaKernel
      real_mult_kernel_host_ip(p->in1,
                               p->in2,
                               p->nelem);

    }
    else{
      // Run the cudaKernel
      real_mult_kernel_host(p->in1,
                            p->in2,
                            p->out,
                            p->nelem);
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ host_v_real_div
int host_v_real_div(CUDA_PLAN_T *p){
  int status=0;

  if(p != NULL){
    if(p->inplace){
      // Run the cudaKernel
      real_div_kernel_host_ip(p->in1,
                              p->in2,
                              p->nelem);

    }
    else{
      // Run the cudaKernel
      real_div_kernel_host(p->in1,
                           p->in2,
                           p->out,
                           p->nelem);
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ host_v_real_add
int host_v_real_add(CUDA_PLAN_T *p){
  int status=0;

  if(p != NULL){
    if(p->inplace){
      // Run the cudaKernel
      real_add_kernel_host_ip(p->in1,
                              p->in2,
                              p->nelem);

    }
    else{
      // Run the cudaKernel
      real_add_kernel_host(p->in1,
                           p->in2,
                           p->out,
                           p->nelem);
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ host_v_real_sub
int host_v_real_sub(CUDA_PLAN_T *p){
  int status=0;

  if(p != NULL){
    if(p->inplace){
      // Run the cudaKernel
      real_sub_kernel_host_ip(p->in1,
                              p->in2,
                              p->nelem);

    }
    else{
      // Run the cudaKernel
      real_sub_kernel_host(p->in1,
                           p->in2,
                           p->out,
                           p->nelem);
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}
//}}}
//}}}

//{{{ CUDA
//{{{ complex
//{{{ kernels
__global__ void cmplx_conj_mult_kernel(cuComplex *a, cuComplex *b, cuComplex *c, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    c[tid] = cuCmulf(cuConjf(a[tid]),b[tid]);
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void cmplx_conj_mult_kernel_ip(cuComplex *a, cuComplex *b, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    b[tid] = cuCmulf(cuConjf(a[tid]),b[tid]);
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void cmplx_mult_kernel(cuComplex *a, cuComplex *b, cuComplex *c, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    c[tid] = cuCmulf(a[tid],b[tid]);
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void cmplx_mult_kernel_ip(cuComplex *a, cuComplex *b, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    b[tid] = cuCmulf(a[tid],b[tid]);
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void cmplx_div_kernel(cuComplex *a, cuComplex *b, cuComplex *c, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    c[tid] = cuCdivf(a[tid],b[tid]);
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void cmplx_div_kernel_ip(cuComplex *a, cuComplex *b, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    b[tid] = cuCdivf(a[tid],b[tid]);
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void cmplx_add_kernel(cuComplex *a, cuComplex *b, cuComplex *c, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    c[tid] = cuCaddf(a[tid],b[tid]);
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void cmplx_add_kernel_ip(cuComplex *a, cuComplex *b, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    b[tid] = cuCaddf(a[tid],b[tid]);
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void cmplx_sub_kernel(cuComplex *a, cuComplex *b, cuComplex *c, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    c[tid] = cuCsubf(a[tid],b[tid]);
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void cmplx_sub_kernel_ip(cuComplex *a, cuComplex *b, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    b[tid] = cuCsubf(a[tid],b[tid]);
    tid += blockDim.x * gridDim.x;
  }
}

//}}}

//{{{ cuda_v_cmplx_conj_mult 
int cuda_v_cmplx_conj_mult(CUDA_PLAN_T *p){
  int status=0;
  int i;

  if(p != NULL){
    if(p->use_zero_copy){
      if(p->inplace){
        // Run the cudaKernel
        cmplx_conj_mult_kernel_ip<<<p->nblocks,p->nthreads>>>((cuComplex *)(p->in1_dev[0]),
                                                              (cuComplex *)(p->in2_dev[0]),
                                                              p->nelem);

      }
      else{
        // Run the cudaKernel
        cmplx_conj_mult_kernel<<<p->nblocks,p->nthreads>>>((cuComplex *)(p->in1_dev[0]),
                                                           (cuComplex *)(p->in2_dev[0]),
                                                           (cuComplex *)(p->out_dev[0]),
                                                           p->nelem);
      }
      // NOTE: This is a key piece in using zero-copy memory
      cudaThreadSynchronize();
    }
    else if(p->use_streams){
      // TODO: Create/destroy these streams in the init function
      cudaStream_t stream[p->num_streams];
      for(i=0;i<p->num_streams;i++) CUDA_ERROR_SETUP(cudaStreamCreate(&stream[i]));

      // NOTE: nblocks is not p->nblocks!!!
      if(p->inplace){
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          cmplx_conj_mult_kernel_ip<<<p->nblocks,p->nthreads,0,stream[0]>>>((cuComplex*)(p->in1_dev[0]),
                                                                            (cuComplex*)(p->in2_dev[0]),
                                                                            p->elem_per_chunk);
          cmplx_conj_mult_kernel_ip<<<p->nblocks,p->nthreads,0,stream[1]>>>((cuComplex*)(p->in1_dev[1]),
                                                                            (cuComplex*)(p->in2_dev[1]),
                                                                            p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+(i*p->elem_per_chunk*2),
                                             p->in2_dev[0],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->in2_dev[1],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i] = cuCmulf(cuConjf(((cuComplex*)(p->in1))[p->nchunks*p->elem_per_chunk+i]),
                                                                             ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i]);
          }
        }

      }
      else {
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          cmplx_conj_mult_kernel<<<p->nblocks,p->nthreads,0,stream[0]>>>((cuComplex*)(p->in1_dev[0]),
                                                                         (cuComplex*)(p->in2_dev[0]),
                                                                         (cuComplex*)(p->out_dev[0]),
                                                                         p->elem_per_chunk);
          cmplx_conj_mult_kernel<<<p->nblocks,p->nthreads,0,stream[1]>>>((cuComplex*)(p->in1_dev[1]),
                                                                         (cuComplex*)(p->in2_dev[1]),
                                                                         (cuComplex*)(p->out_dev[1]),
                                                                         p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+(i*p->elem_per_chunk*2),
                                             p->out_dev[0],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+((i+1)*p->elem_per_chunk*2),
                                             p->out_dev[1],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            ((cuComplex*)(p->out))[p->nchunks*p->elem_per_chunk+i] = cuCmulf(cuConjf(((cuComplex*)(p->in1))[p->nchunks*p->elem_per_chunk+i]),
                                                                             ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i]);
          }
        }
      }
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[0]));
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[1]));

      // TODO: Create/destroy these streams in the init function
      for(i=0;i<p->num_streams;i++) CUDA_ERROR(cudaStreamDestroy(stream[i]));
    }
    else {
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ cuda_v_cmplx_mult
int cuda_v_cmplx_mult(CUDA_PLAN_T *p){
  int status=0;
  int i;

  if(p != NULL){
    if(p->use_zero_copy){
      if(p->inplace){
        // Run the cudaKernel
        cmplx_mult_kernel_ip<<<p->nblocks,p->nthreads>>>((cuComplex *)(p->in1_dev[0]),
                                                              (cuComplex *)(p->in2_dev[0]),
                                                              p->nelem);

      }
      else{
        // Run the cudaKernel
        cmplx_mult_kernel<<<p->nblocks,p->nthreads>>>((cuComplex *)(p->in1_dev[0]),
                                                           (cuComplex *)(p->in2_dev[0]),
                                                           (cuComplex *)(p->out_dev[0]),
                                                           p->nelem);
      }
      // NOTE: This is a key piece in using zero-copy memory
      cudaThreadSynchronize();
    }
    else if(p->use_streams){
      // TODO: Create/destroy these streams in the init function
      cudaStream_t stream[p->num_streams];
      for(i=0;i<p->num_streams;i++) CUDA_ERROR_SETUP(cudaStreamCreate(&stream[i]));

      // NOTE: nblocks is not p->nblocks!!!
      if(p->inplace){
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          cmplx_mult_kernel_ip<<<p->nblocks,p->nthreads,0,stream[0]>>>((cuComplex*)(p->in1_dev[0]),
                                                                            (cuComplex*)(p->in2_dev[0]),
                                                                            p->elem_per_chunk);
          cmplx_mult_kernel_ip<<<p->nblocks,p->nthreads,0,stream[1]>>>((cuComplex*)(p->in1_dev[1]),
                                                                            (cuComplex*)(p->in2_dev[1]),
                                                                            p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+(i*p->elem_per_chunk*2),
                                             p->in2_dev[0],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->in2_dev[1],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i] = cuCmulf(((cuComplex*)(p->in1))[p->nchunks*p->elem_per_chunk+i],
                                                                             ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i]);
          }
        }

      }
      else {
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          cmplx_mult_kernel<<<p->nblocks,p->nthreads,0,stream[0]>>>((cuComplex*)(p->in1_dev[0]),
                                                                         (cuComplex*)(p->in2_dev[0]),
                                                                         (cuComplex*)(p->out_dev[0]),
                                                                         p->elem_per_chunk);
          cmplx_mult_kernel<<<p->nblocks,p->nthreads,0,stream[1]>>>((cuComplex*)(p->in1_dev[1]),
                                                                         (cuComplex*)(p->in2_dev[1]),
                                                                         (cuComplex*)(p->out_dev[1]),
                                                                         p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+(i*p->elem_per_chunk*2),
                                             p->out_dev[0],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+((i+1)*p->elem_per_chunk*2),
                                             p->out_dev[1],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            ((cuComplex*)(p->out))[p->nchunks*p->elem_per_chunk+i] = cuCmulf(((cuComplex*)(p->in1))[p->nchunks*p->elem_per_chunk+i],
                                                                             ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i]);
          }
        }
      }
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[0]));
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[1]));

      // TODO: Create/destroy these streams in the init function
      for(i=0;i<p->num_streams;i++) CUDA_ERROR(cudaStreamDestroy(stream[i]));
    }
    else {
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ cuda_v_cmplx_div
int cuda_v_cmplx_div(CUDA_PLAN_T *p){
  int status=0;
  int i;

  if(p != NULL){
    if(p->use_zero_copy){
      if(p->inplace){
        // Run the cudaKernel
        cmplx_div_kernel_ip<<<p->nblocks,p->nthreads>>>((cuComplex *)(p->in1_dev[0]),
                                                              (cuComplex *)(p->in2_dev[0]),
                                                              p->nelem);

      }
      else{
        // Run the cudaKernel
        cmplx_div_kernel<<<p->nblocks,p->nthreads>>>((cuComplex *)(p->in1_dev[0]),
                                                           (cuComplex *)(p->in2_dev[0]),
                                                           (cuComplex *)(p->out_dev[0]),
                                                           p->nelem);
      }
      // NOTE: This is a key piece in using zero-copy memory
      cudaThreadSynchronize();
    }
    else if(p->use_streams){
      // TODO: Create/destroy these streams in the init function
      cudaStream_t stream[p->num_streams];
      for(i=0;i<p->num_streams;i++) CUDA_ERROR_SETUP(cudaStreamCreate(&stream[i]));

      // NOTE: nblocks is not p->nblocks!!!
      if(p->inplace){
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          cmplx_div_kernel_ip<<<p->nblocks,p->nthreads,0,stream[0]>>>((cuComplex*)(p->in1_dev[0]),
                                                                            (cuComplex*)(p->in2_dev[0]),
                                                                            p->elem_per_chunk);
          cmplx_div_kernel_ip<<<p->nblocks,p->nthreads,0,stream[1]>>>((cuComplex*)(p->in1_dev[1]),
                                                                            (cuComplex*)(p->in2_dev[1]),
                                                                            p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+(i*p->elem_per_chunk*2),
                                             p->in2_dev[0],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->in2_dev[1],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i] = cuCdivf(((cuComplex*)(p->in1))[p->nchunks*p->elem_per_chunk+i],
                                                                             ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i]);
          }
        }

      }
      else {
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          cmplx_div_kernel<<<p->nblocks,p->nthreads,0,stream[0]>>>((cuComplex*)(p->in1_dev[0]),
                                                                         (cuComplex*)(p->in2_dev[0]),
                                                                         (cuComplex*)(p->out_dev[0]),
                                                                         p->elem_per_chunk);
          cmplx_div_kernel<<<p->nblocks,p->nthreads,0,stream[1]>>>((cuComplex*)(p->in1_dev[1]),
                                                                         (cuComplex*)(p->in2_dev[1]),
                                                                         (cuComplex*)(p->out_dev[1]),
                                                                         p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+(i*p->elem_per_chunk*2),
                                             p->out_dev[0],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+((i+1)*p->elem_per_chunk*2),
                                             p->out_dev[1],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            ((cuComplex*)(p->out))[p->nchunks*p->elem_per_chunk+i] = cuCdivf(((cuComplex*)(p->in1))[p->nchunks*p->elem_per_chunk+i],
                                                                             ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i]);
          }
        }
      }
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[0]));
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[1]));

      // TODO: Create/destroy these streams in the init function
      for(i=0;i<p->num_streams;i++) CUDA_ERROR(cudaStreamDestroy(stream[i]));
    }
    else {
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ cuda_v_cmplx_add
int cuda_v_cmplx_add(CUDA_PLAN_T *p){
  int status=0;
  int i;

  if(p != NULL){
    if(p->use_zero_copy){
      if(p->inplace){
        // Run the cudaKernel
        cmplx_add_kernel_ip<<<p->nblocks,p->nthreads>>>((cuComplex *)(p->in1_dev[0]),
                                                              (cuComplex *)(p->in2_dev[0]),
                                                              p->nelem);

      }
      else{
        // Run the cudaKernel
        cmplx_add_kernel<<<p->nblocks,p->nthreads>>>((cuComplex *)(p->in1_dev[0]),
                                                           (cuComplex *)(p->in2_dev[0]),
                                                           (cuComplex *)(p->out_dev[0]),
                                                           p->nelem);
      }
      // NOTE: This is a key piece in using zero-copy memory
      cudaThreadSynchronize();
    }
    else if(p->use_streams){
      // TODO: Create/destroy these streams in the init function
      cudaStream_t stream[p->num_streams];
      for(i=0;i<p->num_streams;i++) CUDA_ERROR_SETUP(cudaStreamCreate(&stream[i]));

      // NOTE: nblocks is not p->nblocks!!!
      if(p->inplace){
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          cmplx_add_kernel_ip<<<p->nblocks,p->nthreads,0,stream[0]>>>((cuComplex*)(p->in1_dev[0]),
                                                                            (cuComplex*)(p->in2_dev[0]),
                                                                            p->elem_per_chunk);
          cmplx_add_kernel_ip<<<p->nblocks,p->nthreads,0,stream[1]>>>((cuComplex*)(p->in1_dev[1]),
                                                                            (cuComplex*)(p->in2_dev[1]),
                                                                            p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+(i*p->elem_per_chunk*2),
                                             p->in2_dev[0],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->in2_dev[1],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i] = cuCaddf(((cuComplex*)(p->in1))[p->nchunks*p->elem_per_chunk+i],
                                                                             ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i]);
          }
        }

      }
      else {
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          cmplx_add_kernel<<<p->nblocks,p->nthreads,0,stream[0]>>>((cuComplex*)(p->in1_dev[0]),
                                                                         (cuComplex*)(p->in2_dev[0]),
                                                                         (cuComplex*)(p->out_dev[0]),
                                                                         p->elem_per_chunk);
          cmplx_add_kernel<<<p->nblocks,p->nthreads,0,stream[1]>>>((cuComplex*)(p->in1_dev[1]),
                                                                         (cuComplex*)(p->in2_dev[1]),
                                                                         (cuComplex*)(p->out_dev[1]),
                                                                         p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+(i*p->elem_per_chunk*2),
                                             p->out_dev[0],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+((i+1)*p->elem_per_chunk*2),
                                             p->out_dev[1],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            ((cuComplex*)(p->out))[p->nchunks*p->elem_per_chunk+i] = cuCaddf(((cuComplex*)(p->in1))[p->nchunks*p->elem_per_chunk+i],
                                                                             ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i]);
          }
        }
      }
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[0]));
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[1]));

      // TODO: Create/destroy these streams in the init function
      for(i=0;i<p->num_streams;i++) CUDA_ERROR(cudaStreamDestroy(stream[i]));
    }
    else {
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ cuda_v_cmplx_sub
int cuda_v_cmplx_sub(CUDA_PLAN_T *p){
  int status=0;
  int i;

  if(p != NULL){
    if(p->use_zero_copy){
      if(p->inplace){
        // Run the cudaKernel
        cmplx_sub_kernel_ip<<<p->nblocks,p->nthreads>>>((cuComplex *)(p->in1_dev[0]),
                                                              (cuComplex *)(p->in2_dev[0]),
                                                              p->nelem);

      }
      else{
        // Run the cudaKernel
        cmplx_sub_kernel<<<p->nblocks,p->nthreads>>>((cuComplex *)(p->in1_dev[0]),
                                                           (cuComplex *)(p->in2_dev[0]),
                                                           (cuComplex *)(p->out_dev[0]),
                                                           p->nelem);
      }
      // NOTE: This is a key piece in using zero-copy memory
      cudaThreadSynchronize();
    }
    else if(p->use_streams){
      // TODO: Create/destroy these streams in the init function
      cudaStream_t stream[p->num_streams];
      for(i=0;i<p->num_streams;i++) CUDA_ERROR_SETUP(cudaStreamCreate(&stream[i]));

      // NOTE: nblocks is not p->nblocks!!!
      if(p->inplace){
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          cmplx_sub_kernel_ip<<<p->nblocks,p->nthreads,0,stream[0]>>>((cuComplex*)(p->in1_dev[0]),
                                                                            (cuComplex*)(p->in2_dev[0]),
                                                                            p->elem_per_chunk);
          cmplx_sub_kernel_ip<<<p->nblocks,p->nthreads,0,stream[1]>>>((cuComplex*)(p->in1_dev[1]),
                                                                            (cuComplex*)(p->in2_dev[1]),
                                                                            p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+(i*p->elem_per_chunk*2),
                                             p->in2_dev[0],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->in2_dev[1],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i] = cuCsubf(((cuComplex*)(p->in1))[p->nchunks*p->elem_per_chunk+i],
                                                                             ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i]);
          }
        }

      }
      else {
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk*2),
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          cmplx_sub_kernel<<<p->nblocks,p->nthreads,0,stream[0]>>>((cuComplex*)(p->in1_dev[0]),
                                                                         (cuComplex*)(p->in2_dev[0]),
                                                                         (cuComplex*)(p->out_dev[0]),
                                                                         p->elem_per_chunk);
          cmplx_sub_kernel<<<p->nblocks,p->nthreads,0,stream[1]>>>((cuComplex*)(p->in1_dev[1]),
                                                                         (cuComplex*)(p->in2_dev[1]),
                                                                         (cuComplex*)(p->out_dev[1]),
                                                                         p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+(i*p->elem_per_chunk*2),
                                             p->out_dev[0],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+((i+1)*p->elem_per_chunk*2),
                                             p->out_dev[1],
                                             p->elem_per_chunk*sizeof(cuComplex),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            ((cuComplex*)(p->out))[p->nchunks*p->elem_per_chunk+i] = cuCsubf(((cuComplex*)(p->in1))[p->nchunks*p->elem_per_chunk+i],
                                                                             ((cuComplex*)(p->in2))[p->nchunks*p->elem_per_chunk+i]);
          }
        }
      }
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[0]));
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[1]));

      // TODO: Create/destroy these streams in the init function
      for(i=0;i<p->num_streams;i++) CUDA_ERROR(cudaStreamDestroy(stream[i]));
    }
    else {
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}
//}}}

//{{{ real
//{{{ kernels
__global__ void real_mult_kernel(float *a, float *b, float *c, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    c[tid] = a[tid]*b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void real_mult_kernel_ip(float *a, float *b, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    b[tid] = a[tid]*b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void real_div_kernel(float *a, float *b, float *c, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    c[tid] = a[tid]/b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void real_div_kernel_ip(float *a, float *b, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    b[tid] = a[tid]/b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void real_add_kernel(float *a, float *b, float *c, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    c[tid] = a[tid]+b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void real_add_kernel_ip(float *a, float *b, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    b[tid] = a[tid]+b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void real_sub_kernel(float *a, float *b, float *c, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    c[tid] = a[tid]-b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void real_sub_kernel_ip(float *a, float *b, long N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < N){
    b[tid] = a[tid]-b[tid];
    tid += blockDim.x * gridDim.x;
  }
}
//}}}

//{{{ cuda_v_real_mul
int cuda_v_real_mult(CUDA_PLAN_T *p){
  int status=0;
  int i;

  if(p != NULL){
    if(p->use_zero_copy){
      if(p->inplace){
        // Run the cudaKernel
        real_mult_kernel_ip<<<p->nblocks,p->nthreads>>>(p->in1_dev[0],p->in2_dev[0],p->nelem);
      }
      else{
        // Run the cudaKernel
        real_mult_kernel<<<p->nblocks,p->nthreads>>>(p->in1_dev[0],p->in2_dev[0],p->out_dev[0],p->nelem);
      }
      // NOTE: This is a key piece in using zero-copy memory
      cudaThreadSynchronize();
    }
    else if(p->use_streams){
      // TODO: Create/destroy these streams in the init function
      cudaStream_t stream[p->num_streams];
      for(i=0;i<p->num_streams;i++) CUDA_ERROR_SETUP(cudaStreamCreate(&stream[i]));

      // NOTE: nblocks is not p->nblocks!!!
      if(p->inplace){
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          real_mult_kernel_ip<<<p->nblocks,p->nthreads,0,stream[0]>>>(p->in1_dev[0],p->in2_dev[0],p->elem_per_chunk);
          real_mult_kernel_ip<<<p->nblocks,p->nthreads,0,stream[1]>>>(p->in1_dev[1],p->in2_dev[1],p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+(i*p->elem_per_chunk),
                                             p->in2_dev[0],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+((i+1)*p->elem_per_chunk),
                                             p->in2_dev[1],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            p->in2[p->nchunks*p->elem_per_chunk+i] = p->in1[p->nchunks*p->elem_per_chunk+i]*p->in2[p->nchunks*p->elem_per_chunk+i];
          }
        }
      }
      else {
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          real_mult_kernel<<<p->nblocks,p->nthreads,0,stream[0]>>>(p->in1_dev[0],
                                                                   p->in2_dev[0],
                                                                   p->out_dev[0],
                                                                   p->elem_per_chunk);
          real_mult_kernel<<<p->nblocks,p->nthreads,0,stream[1]>>>(p->in1_dev[1],
                                                                   p->in2_dev[1],
                                                                   p->out_dev[1],
                                                                   p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+(i*p->elem_per_chunk),
                                             p->out_dev[0],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+((i+1)*p->elem_per_chunk),
                                             p->out_dev[1],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }

        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            p->out[p->nchunks*p->elem_per_chunk+i] = p->in1[p->nchunks*p->elem_per_chunk+i]*p->in2[p->nchunks*p->elem_per_chunk+i];
          }
        }
      }
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[0]));
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[1]));

      // TODO: Create/destroy these streams in the init function
      for(i=0;i<p->num_streams;i++) CUDA_ERROR(cudaStreamDestroy(stream[i]));
    }
    else {
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ cuda_v_real_div
int cuda_v_real_div(CUDA_PLAN_T *p){
  int status=0;
  int i;

  if(p != NULL){
    if(p->use_zero_copy){
      if(p->inplace){
        // Run the cudaKernel
        real_div_kernel_ip<<<p->nblocks,p->nthreads>>>(p->in1_dev[0],
                                                       p->in2_dev[0],
                                                       p->nelem);

      }
      else{
        // Run the cudaKernel
        real_div_kernel<<<p->nblocks,p->nthreads>>>(p->in1_dev[0],
                                                    p->in2_dev[0],
                                                    p->out_dev[0],
                                                    p->nelem);
      }
      // NOTE: This is a key piece in using zero-copy memory
      cudaThreadSynchronize();
    }
    else if(p->use_streams){
      // TODO: Create/destroy these streams in the init function
      cudaStream_t stream[p->num_streams];
      for(i=0;i<p->num_streams;i++) CUDA_ERROR_SETUP(cudaStreamCreate(&stream[i]));

      // NOTE: nblocks is not p->nblocks!!!
      if(p->inplace){
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          real_div_kernel_ip<<<p->nblocks,p->nthreads,0,stream[0]>>>(p->in1_dev[0],
                                                                     p->in2_dev[0],
                                                                     p->elem_per_chunk);
          real_div_kernel_ip<<<p->nblocks,p->nthreads,0,stream[1]>>>(p->in1_dev[1],
                                                                     p->in2_dev[1],
                                                                     p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+(i*p->elem_per_chunk),
                                             p->in2_dev[0],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+((i+1)*p->elem_per_chunk),
                                             p->in2_dev[1],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }

        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            p->in2[p->nchunks*p->elem_per_chunk+i] = p->in1[p->nchunks*p->elem_per_chunk+i]/p->in2[p->nchunks*p->elem_per_chunk+i];
          }
        }

      }
      else {
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          real_div_kernel<<<p->nblocks,p->nthreads,0,stream[0]>>>(p->in1_dev[0],
                                                                  p->in2_dev[0],
                                                                  p->out_dev[0],
                                                                  p->elem_per_chunk);
          real_div_kernel<<<p->nblocks,p->nthreads,0,stream[1]>>>(p->in1_dev[1],
                                                                  p->in2_dev[1],
                                                                  p->out_dev[1],
                                                                  p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+(i*p->elem_per_chunk),
                                             p->out_dev[0],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+((i+1)*p->elem_per_chunk),
                                             p->out_dev[1],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            p->out[p->nchunks*p->elem_per_chunk+i] = p->in1[p->nchunks*p->elem_per_chunk+i]/p->in2[p->nchunks*p->elem_per_chunk+i];
          }
        }
      }
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[0]));
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[1]));

      // TODO: Create/destroy these streams in the init function
      for(i=0;i<p->num_streams;i++) CUDA_ERROR(cudaStreamDestroy(stream[i]));
    }
    else {
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ cuda_v_real_add
int cuda_v_real_add(CUDA_PLAN_T *p){
  int status=0;
  int i;

  if(p != NULL){
    if(p->use_zero_copy){
      if(p->inplace){
        // Run the cudaKernel
        real_add_kernel_ip<<<p->nblocks,p->nthreads>>>(p->in1_dev[0],
                                                       p->in2_dev[0],
                                                       p->nelem);

      }
      else{
        // Run the cudaKernel
        real_add_kernel<<<p->nblocks,p->nthreads>>>(p->in1_dev[0],
                                                    p->in2_dev[0],
                                                    p->out_dev[0],
                                                    p->nelem);
      }
      // NOTE: This is a key piece in using zero-copy memory
      cudaThreadSynchronize();
    }
    else if(p->use_streams){
      // TODO: Create/destroy these streams in the init function
      cudaStream_t stream[p->num_streams];
      for(i=0;i<p->num_streams;i++) CUDA_ERROR_SETUP(cudaStreamCreate(&stream[i]));

      // NOTE: nblocks is not p->nblocks!!!
      if(p->inplace){
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          real_add_kernel_ip<<<p->nblocks,p->nthreads,0,stream[0]>>>(p->in1_dev[0],
                                                                     p->in2_dev[0],
                                                                     p->elem_per_chunk);
          real_add_kernel_ip<<<p->nblocks,p->nthreads,0,stream[1]>>>(p->in1_dev[1],
                                                                     p->in2_dev[1],
                                                                     p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+(i*p->elem_per_chunk),
                                             p->in2_dev[0],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+((i+1)*p->elem_per_chunk),
                                             p->in2_dev[1],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            p->in2[p->nchunks*p->elem_per_chunk+i] = p->in1[p->nchunks*p->elem_per_chunk+i]+p->in2[p->nchunks*p->elem_per_chunk+i];
          }
        }

      }
      else {
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          real_add_kernel<<<p->nblocks,p->nthreads,0,stream[0]>>>(p->in1_dev[0],
                                                                  p->in2_dev[0],
                                                                  p->out_dev[0],
                                                                  p->elem_per_chunk);
          real_add_kernel<<<p->nblocks,p->nthreads,0,stream[1]>>>(p->in1_dev[1],
                                                                  p->in2_dev[1],
                                                                  p->out_dev[1],
                                                                  p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+(i*p->elem_per_chunk),
                                             p->out_dev[0],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+((i+1)*p->elem_per_chunk),
                                             p->out_dev[1],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            p->out[p->nchunks*p->elem_per_chunk+i] = p->in1[p->nchunks*p->elem_per_chunk+i]+p->in2[p->nchunks*p->elem_per_chunk+i];
          }
        }
      }
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[0]));
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[1]));

      // TODO: Create/destroy these streams in the init function
      for(i=0;i<p->num_streams;i++) CUDA_ERROR(cudaStreamDestroy(stream[i]));
    }
    else {
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}

//{{{ cuda_v_real_sub
int cuda_v_real_sub(CUDA_PLAN_T *p){
  int status=0;
  int i;

  if(p != NULL){
    if(p->use_zero_copy){
      if(p->inplace){
        // Run the cudaKernel
        real_sub_kernel_ip<<<p->nblocks,p->nthreads>>>(p->in1_dev[0],
                                                       p->in2_dev[0],
                                                       p->nelem);

      }
      else{
        // Run the cudaKernel
        real_sub_kernel<<<p->nblocks,p->nthreads>>>(p->in1_dev[0],
                                                    p->in2_dev[0],
                                                    p->out_dev[0],
                                                    p->nelem);
      }
      // NOTE: This is a key piece in using zero-copy memory
      cudaThreadSynchronize();
    }
    else if(p->use_streams){
      // TODO: Create/destroy these streams in the init function
      cudaStream_t stream[p->num_streams];
      for(i=0;i<p->num_streams;i++) CUDA_ERROR_SETUP(cudaStreamCreate(&stream[i]));

      // NOTE: nblocks is not p->nblocks!!!
      if(p->inplace){
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          real_sub_kernel_ip<<<p->nblocks,p->nthreads,0,stream[0]>>>(p->in1_dev[0],
                                                                     p->in2_dev[0],
                                                                     p->elem_per_chunk);
          real_sub_kernel_ip<<<p->nblocks,p->nthreads,0,stream[1]>>>(p->in1_dev[1],
                                                                     p->in2_dev[1],
                                                                     p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+(i*p->elem_per_chunk),
                                             p->in2_dev[0],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2+((i+1)*p->elem_per_chunk),
                                             p->in2_dev[1],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            p->in2[p->nchunks*p->elem_per_chunk+i] = p->in1[p->nchunks*p->elem_per_chunk+i]-p->in2[p->nchunks*p->elem_per_chunk+i];
          }
        }

      }
      else {
        for(i=0;i<p->nchunks && !cuda_runtime_failed;i+=p->num_streams){
          // Copy the host memory to the device
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[0],
                                             p->in1+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in1_dev[1],
                                             p->in1+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[0],
                                             p->in2+(i*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->in2_dev[1],
                                             p->in2+((i+1)*p->elem_per_chunk),
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             stream[1]));

          // Run the cudaKernel
          real_sub_kernel<<<p->nblocks,p->nthreads,0,stream[0]>>>(p->in1_dev[0],
                                                                  p->in2_dev[0],
                                                                  p->out_dev[0],
                                                                  p->elem_per_chunk);
          real_sub_kernel<<<p->nblocks,p->nthreads,0,stream[1]>>>(p->in1_dev[1],
                                                                  p->in2_dev[1],
                                                                  p->out_dev[1],
                                                                  p->elem_per_chunk);

          // Copy the output buffer to the host
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+(i*p->elem_per_chunk),
                                             p->out_dev[0],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[0]));
          CUDA_ERROR_RUNTIME(cudaMemcpyAsync(p->out+((i+1)*p->elem_per_chunk),
                                             p->out_dev[1],
                                             p->elem_per_chunk*sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream[1]));

        }
        // Handle leftover
        if(p->elem_leftover > 0){
          for(i=0;i<p->elem_leftover;i++){
            p->out[p->nchunks*p->elem_per_chunk+i] = p->in1[p->nchunks*p->elem_per_chunk+i]-p->in2[p->nchunks*p->elem_per_chunk+i];
          }
        }
      }
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[0]));
      CUDA_ERROR_RUNTIME(cudaStreamSynchronize(stream[1]));

      // TODO: Create/destroy these streams in the init function
      for(i=0;i<p->num_streams;i++) CUDA_ERROR(cudaStreamDestroy(stream[i]));
    }
    else {
    }
  }
  else {
    ERROR("Invalid plan.\n");
    status=1;
  }

  return status;
}
//}}}
//}}}
//}}}
