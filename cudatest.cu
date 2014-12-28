////////////////////////////////////////////////////////////////
//
// Written to test the functionality of the cudafunc library
//
////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <getopt.h>
#include <cudafunc.h>

#define DEFAULT_ELEM 16384

//{{{ command line parsing
typedef struct {
  int verbose;
  long nelem;
  int nblocks;
  int nthreads;
} OPTIONS_T;

void print_usage(){
  printf("cudatest [OPTION] ...\n");
  printf("\n\033[1mDESCRIPTION\033[0m\n");
  printf("\tRuns a series of pass/fail tests on an available GPU.\n");
  printf("\n\033[1mOPTION\033[0m\n");
  printf("\n\t-b, --blocks=N\n");
  printf("\t\tNumber of processing blocks.\n");
  printf("\n\t-h, --help\n");
  printf("\t\tWhat you need.\n");
  printf("\n\t-n, --nelem=SIZE\n");
  printf("\t\tNumber of elements to process\n");
  printf("\n\t-t, --threads=N\n");
  printf("\t\tNumber of threads per block.\n");
  printf("\n\t-v, --verbose=LEVEL\n");
  printf("\t\tVerbosity level.\n");
  return;
}

int parse_opts(int argc, char **argv, OPTIONS_T *op){
  int opt= 0;
  int rtn=0;
  long tc;

  //Specifying the expected options
  //The two options l and b expect numbers as argument
  static struct option long_options[] = {
      {"verbose", optional_argument, 0,'v'},
      {"nelem", required_argument, 0,'n'},
      {"help", no_argument, 0,'h'},
      {"threads", required_argument, 0,'t'},
      {"blocks", required_argument, 0,'b'},
      {0,0,0,0}
  };

  // Default values
  op->verbose=0;
  op->nelem=DEFAULT_ELEM;
  op->nthreads=0;
  op->nblocks=0;
  int long_index =0;
  while ((opt = getopt_long(argc, argv,"hv::n:b:t:", long_options, &long_index )) != -1) {
    if(opt == 'v'){
      op->verbose = 1;
    }
    if(opt == 'n'){
      char *eptr=NULL;
      tc = strtol(optarg, &eptr, 10);
      
      if(tc == 0 && eptr != NULL){
        printf("Invalid value '%s' passed to --nelem\n",optarg);
        rtn=-1;
        break;
      } else {
        if(tc <= 0){
          printf("Invalid number of elements %d given\n",tc);
          rtn=-1;
          break;
        }
        else op->nelem=tc;
      }
    }
    else if(opt == 't'){
      char *eptr=NULL;
      tc = strtol(optarg, &eptr, 10);
      
      if(tc == 0 && eptr != NULL){
        printf("Invalid value '%s' passed to --threads\n",optarg);
        rtn=-1;
        break;
      } else {
        if(tc <= 0){
          printf("Invalid number of threads %d given\n",tc);
          rtn=-1;
          break;
        }
        else op->nthreads=tc;
      }
    }
    else if(opt == 'b'){
      char *eptr=NULL;
      tc = strtol(optarg, &eptr, 10);
      
      if(tc == 0 && eptr != NULL){
        printf("Invalid value '%s' passed to --blocks\n",optarg);
        rtn=-1;
        break;
      } else {
        if(tc <= 0){
          printf("Invalid number of blocks %d given\n",tc);
          rtn=-1;
          break;
        }
        else op->nblocks=tc;
      }
    }

    else if(opt == 'h'){
      print_usage();
      rtn=-1;
      break;
    }
    else if(opt == '?'){
      // An unrecognized option was on the command line. Most applications
      // fall off here and drop execution while printing the usage statement.
      // We're going to print a warning and continue execution.
      printf("Unrecognized option found\n");
      
      print_usage();
      rtn=-1;
      break;
    }
  }

  // If no arguments were given then print the help. If no files were given then print an error. In
  // either case flag an exit
  //if(argc <= 1){
  //  print_usage();
  //  rtn=-1;
  //}
  
  return rtn;
}
//}}}

//{{{ fill_cbuffer
void fill_cbuffer(float *buf, long nelem){
  float a = 5.0;
  for(long i=0;i<nelem;i++){
    buf[i*2]=((float)rand()/(float)(RAND_MAX)) * a;
    buf[i*2+1]=((float)rand()/(float)(RAND_MAX)) * a;
  }
}
//}}}

//{{{ check_stats
void check_stats(double *buf, long nelem, int verbose, double *avg_err, double *var, double *std, double *max_err, long *max_i){
  //double avg_err,var,std,max_err;
  //long max_i=0;
  *avg_err=0.0;
  *max_err=0.0;
  *std=0.0;
  *var=0.0;
  *max_i=0;

  // There are going to be errors between the CPU implementation and the GPU
  // implementation due to the FMA's available on the GPU giving more accurate
  // values.
  // Get the PDF of the error
  for(long i=0;i<nelem;i++){ 
    *avg_err += buf[i];
    if(fabs(buf[i]) > *max_err) {
      *max_err = fabs(buf[i]);
      *max_i=i;
    }
  }
  *avg_err /= (double)nelem;

  for(long i=0;i<nelem;i++){
    *var += (buf[i]-*avg_err)*(buf[i]-*avg_err);
  }
  
  *var /= (double)(nelem-1);
  *std = sqrt(*var);

  return;
}
//}}}

//{{{ run_test_cmplx
int run_test_cmplx(int(*cuda_func)(CUDA_PLAN_T*),
                   int(*host_func)(CUDA_PLAN_T*),
                   float *check_v,
                   double *zero_chk,
                   float *in1,
                   float *in2,
                   long nelem,
                   int nblocks,
                   int nthreads,
                   int verbose){
  int status=0;
  double avg_err,max_err,std,var;
  long max_i;
  timespec ts,te;
  double cudaTime,hostTime;
  CUDA_PLAN_T *p=NULL;

  //{{{ streams
  printf("\tStreams...");
  p = cuda_plan_init(nelem,-1,nblocks,nthreads,CUDA_COMPLEX,verbose);
  show_plan(p);

  memcpy(p->in1,in1,nelem*sizeof(float)*2);
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  clock_gettime(CLOCK_MONOTONIC,&ts);

  cuda_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  cudaTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Copy the result
  memcpy(check_v,p->out,nelem*sizeof(float)*2);

  clock_gettime(CLOCK_MONOTONIC,&ts);

  // Run the host equivalent
  host_func(p); 

  clock_gettime(CLOCK_MONOTONIC,&te);
  hostTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem*2;i++) zero_chk[i] = (double)(check_v[i] - p->out[i]);

  check_stats(zero_chk,nelem*2,verbose,&avg_err,&var,&std,&max_err,&max_i);

  if(fabs(avg_err) < 1e-7) printf("\t\t\033[32mPASSED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
  else {
    printf("\t\t\033[31mFAILED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
    WARN("Error avg= %.12f, std.= %.12f, var.= %.12f, max= %.12f (%ld)\n",avg_err,std,var,max_err,max_i);
    status=1;
  }

  cuda_plan_destroy(p);
  //}}}

  //{{{ streams in-place
  printf("\tStreams in-place...");
  p = cuda_plan_init(nelem,-1,nblocks,nthreads,CUDA_COMPLEX|CUDA_INPLACE,verbose);
  show_plan(p);

  memcpy(p->in1,in1,nelem*sizeof(float)*2);
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  clock_gettime(CLOCK_MONOTONIC,&ts);

  cuda_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  cudaTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Copy the result
  memcpy(check_v,p->in2,nelem*sizeof(float)*2);

  // Re-fill in2
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  clock_gettime(CLOCK_MONOTONIC,&ts);

  // Run the host equivalent
  host_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  hostTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem*2;i++) zero_chk[i] = (double)(check_v[i] - p->in2[i]);

  check_stats(zero_chk,nelem*2,verbose,&avg_err,&var,&std,&max_err,&max_i);

  if(fabs(avg_err) < 1e-7) printf("\t\033[32mPASSED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
  else {
    printf("\t\033[31mFAILED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
    WARN("Error avg= %.12f, std.= %.12f, var.= %.12f, max= %.12f (%ld)\n",avg_err,std,var,max_err,max_i);
    status=1;
  }

  cuda_plan_destroy(p);
  //}}}

  //{{{ zero-copy
  printf("\tZero-copy...");
  p = cuda_plan_init(nelem,-1,nblocks,nthreads,CUDA_COMPLEX|CUDA_ZERO_COPY,verbose);
  show_plan(p);

  memcpy(p->in1,in1,nelem*sizeof(float)*2);
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  clock_gettime(CLOCK_MONOTONIC,&ts);

  cuda_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  cudaTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Copy the result
  memcpy(check_v,p->out,nelem*sizeof(float)*2);

  clock_gettime(CLOCK_MONOTONIC,&ts);

  // Run the host equivalent
  host_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  hostTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem*2;i++) zero_chk[i] = (double)(check_v[i] - p->out[i]);

  check_stats(zero_chk,nelem*2,verbose,&avg_err,&var,&std,&max_err,&max_i);

  if(fabs(avg_err) < 1e-7) printf("\t\t\033[32mPASSED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
  else {
    printf("\t\t\033[31mFAILED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
    WARN("Error avg= %.12f, std.= %.12f, var.= %.12f, max= %.12f (%ld)\n",avg_err,std,var,max_err,max_i);
    status=1;
  }

  cuda_plan_destroy(p);
  //}}}

  //{{{ zero-copy in-place
  printf("\tZero-copy in-place...");
  p = cuda_plan_init(nelem,-1,nblocks,nthreads,CUDA_COMPLEX|CUDA_ZERO_COPY|CUDA_INPLACE,verbose);
  show_plan(p);

  memcpy(p->in1,in1,nelem*sizeof(float)*2);
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  clock_gettime(CLOCK_MONOTONIC,&ts);

  cuda_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  cudaTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Copy the result
  memcpy(check_v,p->in2,nelem*sizeof(float)*2);

  // Re-fill in2
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  clock_gettime(CLOCK_MONOTONIC,&ts);

  // Run the host equivalent
  host_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  hostTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem*2;i++) zero_chk[i] = (double)(check_v[i] - p->in2[i]);

  check_stats(zero_chk,nelem*2,verbose,&avg_err,&var,&std,&max_err,&max_i);

  if(fabs(avg_err) < 1e-7) printf("\t\033[32mPASSED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
  else {
    printf("\t\033[31mFAILED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
    WARN("Error avg= %.12f, std.= %.12f, var.= %.12f, max= %.12f (%ld)\n",avg_err,std,var,max_err,max_i);
    status=1;
  }

  cuda_plan_destroy(p);
  //}}}

  return status;
}
//}}}

//{{{ run_test_real
int run_test_real(int(*cuda_func)(CUDA_PLAN_T*),
                  int(*host_func)(CUDA_PLAN_T*),
                  float *check_v,
                  double *zero_chk,
                  float *in1,
                  float *in2,
                  long nelem,
                  int nblocks,
                  int nthreads,
                  int verbose){
  int status=0;
  double avg_err,max_err,std,var;
  long max_i;
  timespec ts,te;
  double cudaTime,hostTime;
  CUDA_PLAN_T *p=NULL;

  //{{{ streams
  printf("\tStreams...");
  p = cuda_plan_init(nelem,-1,nblocks,nthreads,0,verbose);
  show_plan(p);

  memcpy(p->in1,in1,nelem*sizeof(float));
  memcpy(p->in2,in2,nelem*sizeof(float));

  clock_gettime(CLOCK_MONOTONIC,&ts);

  cuda_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  cudaTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Copy the result
  memcpy(check_v,p->out,nelem*sizeof(float));

  clock_gettime(CLOCK_MONOTONIC,&ts);

  // Run the host equivalent
  host_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  hostTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem;i++) zero_chk[i] = (double)(check_v[i] - p->out[i]);

  check_stats(zero_chk,nelem,verbose,&avg_err,&var,&std,&max_err,&max_i);

  if(fabs(avg_err) < 1e-7) printf("\t\t\033[32mPASSED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
  else {
    printf("\t\t\033[31mFAILED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
    WARN("Error avg= %.12f, std.= %.12f, var.= %.12f, max= %.12f (%ld)\n",avg_err,std,var,max_err,max_i);
    status=1;
  }

  cuda_plan_destroy(p);
  //}}}

  //{{{ streams in-place
  printf("\tStreams in-place...");
  p = cuda_plan_init(nelem,-1,nblocks,nthreads,CUDA_INPLACE,verbose);
  show_plan(p);

  memcpy(p->in1,in1,nelem*sizeof(float));
  memcpy(p->in2,in2,nelem*sizeof(float));

  clock_gettime(CLOCK_MONOTONIC,&ts);

  cuda_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  cudaTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Copy the result
  memcpy(check_v,p->in2,nelem*sizeof(float));

  // Re-fill in2
  memcpy(p->in2,in2,nelem*sizeof(float));

  clock_gettime(CLOCK_MONOTONIC,&ts);

  // Run the host equivalent
  host_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  hostTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem;i++) zero_chk[i] = (double)(check_v[i] - p->in2[i]);

  check_stats(zero_chk,nelem,verbose,&avg_err,&var,&std,&max_err,&max_i);

  if(fabs(avg_err) < 1e-7) printf("\t\033[32mPASSED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
  else {
    printf("\t\033[31mFAILED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
    WARN("Error avg= %.12f, std.= %.12f, var.= %.12f, max= %.12f (%ld)\n",avg_err,std,var,max_err,max_i);
    status=1;
  }

  cuda_plan_destroy(p);
  //}}}

  //{{{ zero-copy
  printf("\tZero-copy...");
  p = cuda_plan_init(nelem,-1,nblocks,nthreads,CUDA_ZERO_COPY,verbose);
  show_plan(p);

  memcpy(p->in1,in1,nelem*sizeof(float));
  memcpy(p->in2,in2,nelem*sizeof(float));

  clock_gettime(CLOCK_MONOTONIC,&ts);

  cuda_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  cudaTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Copy the result
  memcpy(check_v,p->out,nelem*sizeof(float));

  clock_gettime(CLOCK_MONOTONIC,&ts);

  // Run the host equivalent
  host_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  hostTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem;i++) zero_chk[i] = (double)(check_v[i] - p->out[i]);

  check_stats(zero_chk,nelem,verbose,&avg_err,&var,&std,&max_err,&max_i);

  if(fabs(avg_err) < 1e-7) printf("\t\t\033[32mPASSED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
  else {
    printf("\t\t\033[31mFAILED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
    WARN("Error avg= %.12f, std.= %.12f, var.= %.12f, max= %.12f (%ld)\n",avg_err,std,var,max_err,max_i);
    status=1;
  }

  cuda_plan_destroy(p);
  //}}}

  //{{{ zero-copy in-place
  printf("\tZero-copy in-place...");
  p = cuda_plan_init(nelem,-1,nblocks,nthreads,CUDA_ZERO_COPY|CUDA_INPLACE,verbose);
  show_plan(p);

  memcpy(p->in1,in1,nelem*sizeof(float));
  memcpy(p->in2,in2,nelem*sizeof(float));

  clock_gettime(CLOCK_MONOTONIC,&ts);

  cuda_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  cudaTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Copy the result
  memcpy(check_v,p->in2,nelem*sizeof(float));

  // Re-fill in2
  memcpy(p->in2,in2,nelem*sizeof(float));

  clock_gettime(CLOCK_MONOTONIC,&ts);

  // Run the host equivalent
  host_func(p);

  clock_gettime(CLOCK_MONOTONIC,&te);
  hostTime = (te.tv_sec - ts.tv_sec) + ((te.tv_nsec-ts.tv_nsec)/1e9);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem;i++) zero_chk[i] = (double)(check_v[i] - p->in2[i]);

  check_stats(zero_chk,nelem,verbose,&avg_err,&var,&std,&max_err,&max_i);

  if(fabs(avg_err) < 1e-7) printf("\t\033[32mPASSED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
  else {
    printf("\t\033[31mFAILED\033[0m (cuda= %.12f,host= %.12f)\n",cudaTime,hostTime);
    WARN("Error avg= %.12f, std.= %.12f, var.= %.12f, max= %.12f (%ld)\n",avg_err,std,var,max_err,max_i);
    status=1;
  }

  cuda_plan_destroy(p);
  //}}}

  return status;
}
//}}}


int main(int argc, char **argv){
  int status=0;
  OPTIONS_T opts;
  int ostat = parse_opts(argc,argv,&opts);
  if(ostat != 0) exit(1);
  if(opts.verbose > 0){
    INFO("Checking for CUDA enabled devices\n");
    show_devices();
  }
  //CUDA_PLAN_T *p=NULL;

  long nelem=opts.nelem;

  srand((unsigned int)time(NULL));

  // NOTE: These are all complex operations
  float *check_v = (float *) calloc(nelem,sizeof(float)*2);
  double *zero_chk = (double *) calloc(nelem,sizeof(double)*2);
  float *in1 = (float *) calloc(nelem,sizeof(float)*2);
  float *in2 = (float *) calloc(nelem,sizeof(float)*2);

  // Fill the buffers
  fill_cbuffer(in1,nelem);
  fill_cbuffer(in2,nelem);

//{{{ complex
  //////////////////////////////////////////////////////////////////////
  INFO("Running a CUDA enabled complex conj multiply...\n");
  status = run_test_cmplx(cuda_v_cmplx_conj_mult,host_v_cmplx_conj_mult,check_v,zero_chk,in1,in2,nelem,opts.nblocks,opts.nthreads,opts.verbose);

  INFO("Running a CUDA enabled complex multiply...\n");
  status = run_test_cmplx(cuda_v_cmplx_mult,host_v_cmplx_mult,check_v,zero_chk,in1,in2,nelem,opts.nblocks,opts.nthreads,opts.verbose);

  INFO("Running a CUDA enabled complex divide...\n");
  status = run_test_cmplx(cuda_v_cmplx_div,host_v_cmplx_div,check_v,zero_chk,in1,in2,nelem,opts.nblocks,opts.nthreads,opts.verbose);

  INFO("Running a CUDA enabled complex add...\n");
  status = run_test_cmplx(cuda_v_cmplx_add,host_v_cmplx_add,check_v,zero_chk,in1,in2,nelem,opts.nblocks,opts.nthreads,opts.verbose);

  INFO("Running a CUDA enabled complex subtract...\n");
  status = run_test_cmplx(cuda_v_cmplx_sub,host_v_cmplx_sub,check_v,zero_chk,in1,in2,nelem,opts.nblocks,opts.nthreads,opts.verbose);
  //////////////////////////////////////////////////////////////////////
//}}}

//{{{ real
  INFO("Running a CUDA enabled real multiply...\n");
  status = run_test_real(cuda_v_real_mult,host_v_real_mult,check_v,zero_chk,in1,in2,nelem,opts.nblocks,opts.nthreads,opts.verbose);

  INFO("Running a CUDA enabled real divide...\n");
  status = run_test_real(cuda_v_real_div,host_v_real_div,check_v,zero_chk,in1,in2,nelem,opts.nblocks,opts.nthreads,opts.verbose);

  INFO("Running a CUDA enabled real add...\n");
  status = run_test_real(cuda_v_real_add,host_v_real_add,check_v,zero_chk,in1,in2,nelem,opts.nblocks,opts.nthreads,opts.verbose);

  INFO("Running a CUDA enabled real subtract...\n");
  status = run_test_real(cuda_v_real_sub,host_v_real_sub,check_v,zero_chk,in1,in2,nelem,opts.nblocks,opts.nthreads,opts.verbose);

//}}}

  if(check_v != NULL) free(check_v);
  if(zero_chk != NULL) free(zero_chk);
  if(in1 != NULL) free(in1);
  if(in2 != NULL) free(in2);

  return status;
}
