////////////////////////////////////////////////////////////////
//
// Written to test the functionality of the cudafunc library
//
////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <cudafunc.h>

#define DEFAULT_ELEM 16384

//{{{ command line parsing
typedef struct {
  int verbose;
  long nelem;
} OPTIONS_T;

void print_usage(){
  printf("cudatest [OPTION] ...\n");
  printf("\n\033[1mDESCRIPTION\033[0m\n");
  printf("\tRuns a series of pass/fail tests on an available GPU.\n");
  printf("\n\033[1mOPTION\033[0m\n");
  printf("\n\t-h, --help\n");
  printf("\t\tWhat you need.\n");
  printf("\n\t-n, --nelem=SIZE\n");
  printf("\t\tNumber of elements to process\n");
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
      {0,0,0,0}
  };

  // Default values
  op->verbose=0;
  op->nelem=DEFAULT_ELEM;
  int long_index =0;
  while ((opt = getopt_long(argc, argv,"hv::n:", long_options, &long_index )) != -1) {
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
int check_stats(double *buf, long nelem, int verbose){
  int tolerable=0;
  double avg_err,var,std,max_err;
  long max_i=0;

  // There are going to be errors between the CPU implementation and the GPU
  // implementation due to the FMA's available on the GPU giving more accurate
  // values.
  // Get the PDF of the error
  avg_err=0.0;
  max_err=0.0;
  for(long i=0;i<nelem;i++){ 
    avg_err += buf[i];
    if(fabs(buf[i]) > max_err) {
      max_err = fabs(buf[i]);
      max_i=i;
    }
  }
  avg_err /= (double)nelem;

  var=0.0;
  for(long i=0;i<nelem;i++){
    var += (buf[i]-avg_err)*(buf[i]-avg_err);
  }
  
  var /= (double)(nelem-1);
  std = sqrt(var);

  if(fabs(avg_err) < 1e-7) tolerable=1;
  if(!tolerable || verbose > 0) WARN("Error avg= %.12f, std.= %.12f, var.= %.12f, max= %.12f (%ld)\n",avg_err,std,var,max_err,max_i);

  return tolerable;
}
//}}}

int main(int argc, char **argv){
  OPTIONS_T opts;
  int ostat = parse_opts(argc,argv,&opts);
  if(ostat != 0) exit(1);
  if(opts.verbose > 0){
    INFO("Checking for CUDA enabled devices\n");
    show_devices();
  }
  CUDA_PLAN_T *p=NULL;

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

  //////////////////////////////////////////////////////////////////////
  //{{{ complex multiply
  INFO("Running a CUDA enabled complex conj multiply...");
  if(opts.verbose) printf("\n");
  p = cuda_plan_init(nelem,-1,-1,-1,CUDA_COMPLEX,opts.verbose);
  show_plan(p);

  memcpy(p->in1,in1,nelem*sizeof(float)*2);
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  cuda_v_cmplx_conj_mult(p);

  // Copy the result
  memcpy(check_v,p->out,nelem*sizeof(float)*2);

  // Run the host equivalent
  host_v_cmplx_conj_mult(p);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem*2;i++) zero_chk[i] = (double)(check_v[i] - p->out[i]);

  if(!check_stats(zero_chk,nelem*2,opts.verbose)){
    printf("\033[31mFAILED\033[0m");
    //for(long i=0;i<nelem*2;i++) printf("%d= %.12f\n",i,zero_chk[i]);
  }
  else printf("\033[32mPASSED\033[0m");

  cuda_plan_destroy(p);
  printf("\n");
  //}}}
  //////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////
  //{{{ complex multiply in-place
  INFO("Running a CUDA enabled complex conj multiply in-place...");
  if(opts.verbose) printf("\n");
  p = cuda_plan_init(nelem,-1,-1,-1,CUDA_COMPLEX|CUDA_INPLACE,opts.verbose);
  show_plan(p);

  // Fill the buffers
  memcpy(p->in1,in1,nelem*sizeof(float)*2);
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  cuda_v_cmplx_conj_mult(p);

  // Copy the result
  memcpy(check_v,p->in2,nelem*sizeof(float)*2);

  // Put the data back in place
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  // Run the host equivalent
  host_v_cmplx_conj_mult(p);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem*2;i++) zero_chk[i] = (double)(check_v[i] - p->in2[i]);

  if(!check_stats(zero_chk,nelem*2,opts.verbose)) printf("\033[31mFAILED\033[0m");
  else printf("\033[32mPASSED\033[0m");

  cuda_plan_destroy(p);
  printf("\n");
  //}}}
  //////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////
  //{{{ complex multiply zero-copy
  INFO("Running a CUDA enabled complex conj multiply zero-copy...");
  if(opts.verbose) printf("\n");
  p = cuda_plan_init(nelem,-1,-1,-1,CUDA_COMPLEX|CUDA_ZERO_COPY,opts.verbose);
  show_plan(p);

  // Fill the buffers
  memcpy(p->in1,in1,nelem*sizeof(float)*2);
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  cuda_v_cmplx_conj_mult(p);

  // Copy the result
  memcpy(check_v,p->out,nelem*sizeof(float)*2);

  // Run the host equivalent
  host_v_cmplx_conj_mult(p);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem*2;i++) zero_chk[i] = (double)(check_v[i] - p->out[i]);

  if(!check_stats(zero_chk,nelem*2,opts.verbose)) printf("\033[31mFAILED\033[0m");
  else printf("\033[32mPASSED\033[0m");

  cuda_plan_destroy(p);
  printf("\n");
  //}}}
  //////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////
  //{{{ complex multiply zero-copy in-place
  INFO("Running a CUDA enabled complex conj multiply zero-copy in-place...");
  if(opts.verbose) printf("\n");
  p = cuda_plan_init(nelem,-1,-1,-1,CUDA_COMPLEX|CUDA_ZERO_COPY|CUDA_INPLACE,opts.verbose);
  show_plan(p);

  // Fill the buffers
  memcpy(p->in1,in1,nelem*sizeof(float)*2);
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  cuda_v_cmplx_conj_mult(p);

  // Copy the result
  memcpy(check_v,p->in2,nelem*sizeof(float)*2);

  // Put the data back in place
  memcpy(p->in2,in2,nelem*sizeof(float)*2);

  // Run the host equivalent
  host_v_cmplx_conj_mult(p);

  // Check the result against the CUDA version.
  for(long i=0;i<nelem*2;i++) zero_chk[i] = (double)(check_v[i] - p->in2[i]);

  if(!check_stats(zero_chk,nelem*2,opts.verbose)) printf("\033[31mFAILED\033[0m");
  else printf("\033[32mPASSED\033[0m");

  cuda_plan_destroy(p);
  printf("\n");
  //}}}
  //////////////////////////////////////////////////////////////////////

  if(check_v != NULL) free(check_v);
  if(zero_chk != NULL) free(zero_chk);
  if(in1 != NULL) free(in1);
  if(in2 != NULL) free(in2);

  return 0;
}
