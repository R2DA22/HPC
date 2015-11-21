#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 32

//////////////////////////////////////////////////////
//Funcion que  multiplica palarelamente las matrices//
//////////////////////////////////////////////////////

__global__ void matMultParallel(float* A, float* B, float* C,  int n, int m, int o){
	int row= blockIdx.y*blockDim.y + threadIdx.y;

	int col= blockIdx.x*blockDim.x + threadIdx.x;

	if ((row < n) && (col < o)){
		float pvalue=0;
		
		for(int k=0 ; k < m ; k++ ){
			pvalue+=A[row*m+k]*B[k*o+col];
		}
		C[row*o+col]=pvalue;
	}
	
}

//////////////////////////////////////////////////
//Funcion que  multiplica las matrices con tiling//
//////////////////////////////////////////////////

__global__ void matMultParallelTiled(float* A, float* B, float* C, int n, int m, int o){
  
	__shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float Pvalue=0;
	for (int k = 0; k < (m+TILE_WIDTH-1)/TILE_WIDTH; ++k){
		if ( row < n  &&  (k*TILE_WIDTH + tx) < m){
			Ads[ty][tx] = A[row * m + k*TILE_WIDTH + tx];
		}else{
			Ads[ty][tx] = 0;
		}
		if ((k*TILE_WIDTH + ty) < m && col < o){
			Bds[ty][tx] = B[(k*TILE_WIDTH + ty) * o + col];
		}else{
			Bds[ty][tx] =0;
		}
		__syncthreads();
		for(int k = 0; k < TILE_WIDTH; ++k){
			Pvalue += Ads[ty][k] * Bds[k][tx];
		}
		__syncthreads();
	}
	if (row < n && col < o){
		C[row * o + col] = Pvalue;
	}
}

////////////////////////////////////////////////////////////
//Funcion que  multiplica las matrices de forma secuencial//
////////////////////////////////////////////////////////////

void multMatrixsequential(float *A, float *B, float *C, int n, int m, int o){
	for (int i=0; i<n; i++){
		for (int j=0; j<o; j++){
			float sum=0;
			for (int k=0; k<m; k++){
				sum += A[m*i+k]*B[o*k+j];
			}
			C[o*i+j] = sum;
		}
	}
}

///////////////////////////////////
//Funcion que tiene codigo cuda c//
///////////////////////////////////

int matrixMult( float *A, float *B, float *C,int n, int m ,int o, int flag){
  
	int sizeA=n*m*sizeof(int);
	int sizeB=m*o*sizeof(int);
	int sizeC=n*o*sizeof(int);
  
	float *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A,	sizeA);
	cudaMalloc((void **)&d_B, sizeB);
	cudaMalloc((void **)&d_C,	sizeC);
	
	clock_t t;
	t=clock();
	//Copio los datos al device
	cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
  
	dim3 dimBlock(32.0,32.0); //mayor cantidad de hilos por bloque
	dim3 dimGrid(ceil((float)n/dimBlock.x),ceil((float)n/dimBlock.y));
	// Ejecuto el Kernel (del dispositivo)
	if(flag==1){
		matMultParallelTiled<<<dimGrid,dimBlock>>>(d_A, d_B, d_C,n,m,o);
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
		printf("Multiplicacion paralela con tiling\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
	}else{
	matMultParallel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C,n,m,o);
		cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
		printf("Multiplicacion paralela sin tiling\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
	}
	
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}

/////////////////////////////////////
//Funcion que  imprime las matrices//
/////////////////////////////////////

void PrintMatrix(float *matrix, int n, int m){
	printf("\n");
	for(int i=0; i< n; i++){
		for(int j=0 ; j< m;j++){
			printf("%f ",matrix[i*m+j]);
		}
		printf("\n");
	}
}

//////////////////////////////////////////////////////////
//Funcion que  llena las matrices con valores aleatorios//
//////////////////////////////////////////////////////////

void FillMatrix(float *matrix,int n,int m, int r){
	for(int i=0; i < n*m; i++){
		matrix[i]=rand()% r;
	}
}

//////////////////////////
//Funcion main principal//
//////////////////////////

int main(){
	int n=5;
	int m=6;
	int o=5;
    
	float *A=(float *) malloc(n*m*sizeof(float));
	
	float *B=(float *) malloc(m*o*sizeof(float));
	
	float *C=(float *) malloc(n*o*sizeof(float));

	FillMatrix(A, n, m,10);
	FillMatrix(B, m, o,10);
	printf("==============Tiempos==============\n");
	clock_t t;
	t=clock();
	multMatrixsequential(A,B,C,n,m,o);
	printf("Multiplicacion secuencial\t\t\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
	//PrintMatrix(C,n,o);
	matrixMult(A,B,C,n,m,o,0);
	//PrintMatrix(C,n,o);
	matrixMult(A,B,C,n,m,o,1);
	//PrintMatrix(A,n,m);
	//PrintMatrix(B,m,o);
	//PrintMatrix(C,n,o);
	free(A);
	free(B);
	free(C);	
	
	return 0;

}