#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

#define blockSize 1024

const int  SIZE[] = {512,1024,5000,10000,20000,40000,80000,160000,320000,500000};


__global__ void vecAdd(int *A, int *B, int *C,int n){
	int i=threadIdx.x;
	//int i = blockIdx.x*blockDim.x+threadIdx.x;
	//int i = blockIdx.x
	if(i< n){
		C[i]=A[i]+B[i];
		printf("%i + %i = %i \n",A[i] ,B[i],C[i]);
		
	}


}

int vectorAdd( int *A, int *B, int *C, int n,int j){
	int size = n*sizeof(int);
	int *d_A, *d_B, *d_C;
	int dimGrid = 0;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);
	//Copio los datos al device
	
	

	dimGrid = (int)ceil((float)n/blockSize);
  	//printf("%d\n", dimGrid);
	clock_t t;
	t=clock();
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	// Ejecuto el Kernel (del dispositivo)
	vecAdd<<< dimGrid, blockSize>>>(d_A, d_B, d_C, n);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	printf("El tiempo para %i es de \t: %.8f\n",SIZE[j],(clock()-t)/(double)CLOCKS_PER_SEC);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}


int main(){
	int j;
	int *A;
    int *B;
  	int *C;
	for( j=0; j<1;j++){
    
		A=(int *) malloc(SIZE[j]*sizeof(int));
	
		B=(int *) malloc(SIZE[j]*sizeof(int));
	
		C=(int *) malloc(SIZE[j]*sizeof(int));
		int i;
		for(i=0;i < SIZE[j]; i++){
			A[i]=rand()% 20;
			B[i]=rand()% 20;
		
		}
		vectorAdd(A,B,C,SIZE[j],j);
	}
	free(A);
	free(B);
	free(C);
	return 0;
}