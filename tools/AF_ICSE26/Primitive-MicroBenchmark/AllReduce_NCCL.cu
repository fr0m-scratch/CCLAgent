#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "comm.h"       // <<< pulls in the *internal* comm definitions
#include "device.h"       // <<< or whichever header actually defines `struct ncclCommAndChannels`
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <chrono>         // for std::chrono::high_resolution_clock

#define N_CHUNKS 32

int topo_cclinsight[CCLInsight_CHANNELS][CCLInsight_PEERS];

int CCLInsight_algo;

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

uint64_t rdtsc() {
    uint32_t lo, hi;
    // Inline assembly to read the TSC
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (uint64_t)hi << 32 | lo;
}

int main(int argc, char* argv[])
{

  const char* env_CCLInsight_algo = getenv("NCCL_ALGO");

  const char* env_gauge_mode_var = getenv("GAUGE_MODE");

  const char* env_gauge_iteration_var = getenv("GAUGE_ITERATION");

  const char* env_gauge_nchannels_var = getenv("GAUGE_NCHANNELS");

  const char* env_gauge_buff_size_var = getenv("NCCL_BUFFSIZE");

  const char* env_gauge_output_dir_var = getenv("CCLInsight_OUT_DIR");

  const char* env_gauge_nthreads_var = getenv("NCCL_NTHREADS");

  // Check if environment variables are set
  if (!env_CCLInsight_algo) {
    env_CCLInsight_algo = "unknown_algo";
    printf("unknown Algorithm !!!\n");
  }
  if (!env_gauge_mode_var) env_gauge_mode_var = "unknown_gauge_mode";
  if (!env_gauge_iteration_var) env_gauge_iteration_var = "unknown_gauge_iteration";
  if (!env_gauge_nchannels_var) env_gauge_nchannels_var = "unknown_gauge_nchannels";
  if (!env_gauge_buff_size_var) env_gauge_buff_size_var = "unknown_gauge_buff_size";
  if (!env_gauge_nthreads_var) env_gauge_nthreads_var = "unknown_gauge_nthreads";  
  if (!env_gauge_output_dir_var) {
    env_gauge_output_dir_var = "unknown_gauge_output_dir";
    printf("unknown gauge output dir\n");
  }

  long long size = 1;  // Default size
  const char* env_gauge_size_var = getenv("GAUGE_MESSAGE_SIZE");
  if (env_gauge_size_var != nullptr) {
      size = atoll(env_gauge_size_var) * 1024 / 4;  // Convert from kilobytes to number of floats, assuming the environment variable is in kilobytes
  }

  int myRank, nRanks, localRank = 0;


  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  char filename[256];

  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  ncclUniqueId id;
  float *sendbuff, *recvbuff;
  cudaStream_t s;


  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float))); 
  CUDACHECK(cudaStreamCreate(&s));

  // before ncclCommInitRank(...)
  LogMessage_CCLInsight* d_profBuf = nullptr;
  CUDACHECK(cudaMalloc(&d_profBuf,
    CCLInsight_CHANNELS * sizeof(LogMessage_CCLInsight)));
  CUDACHECK(cudaMemset(d_profBuf, 0,
    CCLInsight_CHANNELS * sizeof(LogMessage_CCLInsight)));
  // d_profBuf now holds the device address of your buffer
  
  //initializing NCCL
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  auto commHost = (ncclComm*)comm;

  // Then we wrote that pointer into NCCL’s device struct:
  uintptr_t base = reinterpret_cast<uintptr_t>(commHost->devComm)
                - offsetof(ncclDevCommAndChannels, comm);
  uintptr_t addr = base + offsetof(ncclDevCommAndChannels, comm.profBuf);
  cudaMemcpy(reinterpret_cast<void*>(addr),
            &d_profBuf,              // the host copy of the pointer value
            sizeof(d_profBuf),
            cudaMemcpyHostToDevice);


  //communicating using NCCL

  cudaEvent_t start, stop;
  float elapsed_time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  CUDACHECK(cudaStreamSynchronize(s));

  cudaEventRecord(start, s);

  std::chrono::time_point<std::chrono::high_resolution_clock> nccl_func_start_time = std::chrono::high_resolution_clock::now(); 

  CUDACHECK(cudaStreamSynchronize(s));

  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
        comm, s));

  CUDACHECK(cudaStreamSynchronize(s));

  cudaEventRecord(stop, s);

  std::chrono::time_point<std::chrono::high_resolution_clock> nccl_func_end_time = std::chrono::high_resolution_clock::now();

  // Wait for the stop event to complete
  cudaEventSynchronize(stop);

  // Calculate elapsed time between events
  cudaEventElapsedTime(&elapsed_time, start, stop);

  // Destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop); 
  
  std::chrono::duration<float, std::milli> nccl_func_time = nccl_func_end_time - nccl_func_start_time; 

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  if (myRank < 16) {
    sprintf(filename, "%s/nccl_allreduce_r-%d.out", env_gauge_output_dir_var, myRank);
    freopen(filename, "a", stdout);
  } else {
    freopen("/dev/null", "w", stdout);
  }

  // allocate a host array large enough for all channels
  auto h_messages_all_channels = (LogMessage_CCLInsight*)malloc(
    CCLInsight_CHANNELS * sizeof(LogMessage_CCLInsight));

  CUDACHECK(cudaMemcpy(
    h_messages_all_channels,    // host pointer
    d_profBuf,     // device pointer you originally malloc’d
    CCLInsight_CHANNELS * sizeof(LogMessage_CCLInsight),
    cudaMemcpyDeviceToHost
  ));

  double gauge_time;

  // now we only profile the channel 0
  LogMessage_CCLInsight* h_messages = &h_messages_all_channels[0];

  // How many entries to loop over?
  size_t iteration = std::min(h_messages->Chunk_numbers+1, (unsigned long long)CCLInsight_MAXLOG);
  
  if (env_CCLInsight_algo && strcasecmp(env_CCLInsight_algo, "TREE") == 0) { // now the algo is TREE
    if (topo_cclinsight[0][0] == -1) {
      printf("message size(%s)_nchannels(%s)_nthreads(%s)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_iteration_var, elapsed_time);
      printf("message size(%s)_nchannels(%s)_nthreads(%s)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_iteration_var, nccl_func_time.count());
      for (size_t i = 0; i < iteration; ++i) { 
        gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[0][i]) / GAUGE_GPU_FREQUENCY;
        printf("rrcs_mode(%s)_nchannels(%s)_nthreads(%s)_buff size(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_buff_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
      }
    } else if (topo_cclinsight[0][1] == -1) {
      printf("message size(%s)_nchannels(%s)_nthreads(%s)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_iteration_var, elapsed_time);
      printf("message size(%s)_nchannels(%s)_nthreads(%s)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_iteration_var, nccl_func_time.count());
      for (size_t i = 0; i < iteration; ++i) { 
        gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[0][i]) / GAUGE_GPU_FREQUENCY;
        printf("send_mode(%s)_nchannels(%s)_nthreads(%s)_buff size(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_buff_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
        gauge_time = static_cast<double>(h_messages->timeValue[3][i] - h_messages->timeValue[2][i]) / GAUGE_GPU_FREQUENCY;
        printf("recv_mode(%s)_nchannels(%s)_nthreads(%s)_buff size(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_buff_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
      }
    } else {
      printf("message size(%s)_nchannels(%s)_nthreads(%s)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_iteration_var, elapsed_time);
      printf("message size(%s)_nchannels(%s)_nthreads(%s)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_iteration_var, nccl_func_time.count());
      for (size_t i = 0; i < iteration; ++i) { 
        gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[0][i]) / GAUGE_GPU_FREQUENCY;
        printf("rrs_mode(%s)_nchannels(%s)_nthreads(%s)_buff size(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_buff_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
        gauge_time = static_cast<double>(h_messages->timeValue[3][i] - h_messages->timeValue[2][i]) / GAUGE_GPU_FREQUENCY;
        printf("rcs_mode(%s)_nchannels(%s)_nthreads(%s)_buff size(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_buff_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
      }
    }
  } else if (env_CCLInsight_algo && strcasecmp(env_CCLInsight_algo, "RING") == 0) { // now the algo is RING
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_iteration_var, elapsed_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_iteration_var, nccl_func_time.count());
    for (size_t i = 0; i < iteration; ++i) { 
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[0][i]) / GAUGE_GPU_FREQUENCY;
      printf("send_mode(%s)_nchannels(%s)_nthreads(%s)_buff size(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_buff_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[3][i] - h_messages->timeValue[2][i]) / GAUGE_GPU_FREQUENCY;
      printf("rrs_mode(%s)_nchannels(%s)_nthreads(%s)_buff size(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_buff_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[5][i] - h_messages->timeValue[4][i]) / GAUGE_GPU_FREQUENCY;
      printf("rrcs_mode(%s)_nchannels(%s)_nthreads(%s)_buff size(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_buff_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[7][i] - h_messages->timeValue[6][i]) / GAUGE_GPU_FREQUENCY;
      printf("rcs_mode(%s)_nchannels(%s)_nthreads(%s)_buff size(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_buff_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[9][i] - h_messages->timeValue[8][i]) / GAUGE_GPU_FREQUENCY;
      printf("recv_mode(%s)_nchannels(%s)_nthreads(%s)_buff size(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_buff_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
    }
  }

  // Free the host buffer
  free(h_messages_all_channels);


  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));

  // free profile buffer
  CUDACHECK(cudaFree(d_profBuf));


  //finalizing NCCL
  ncclCommDestroy(comm);

  //finalizing MPI
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}