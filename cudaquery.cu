////////////////////////////////////////////////////////////////
//
// Written to serve as a quick query utility
//
////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <cudafunc.h>

int main(int argc, char **argv){
  int status=0;

  INFO("Checking for CUDA enabled devices\n");
  show_devices();

  return status;
}
