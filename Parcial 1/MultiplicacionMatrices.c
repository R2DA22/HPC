#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 32

//////////////////////////////////////////////////////
//Funcion que  multiplica palarelamente las matrices//
//////////////////////////////////////////////////////

__global__ void matMultParallel(int* A, int* B, int* C,  int n, int m, int o){
	int row= blockIdx.y*blockDim.y + threadIdx.y;

	int col= blockIdx.x*blockDim.x + threadIdx.x;

	if ((row < n) && (col < o)){
		int pvalue=0;
		
		for(int k=0 ; k < m ; k++ ){
			pvalue+=A[row*m+k]*B[k*o+col];
		}
		C[row*o+col]=pvalue;
	}
	
}

//////////////////////////////////////////////////
//Funcion que  multiplica las matrices con tiled//
//////////////////////////////////////////////////

__global__ void matMultParallelTiled(int* A, int* B, int* C, int n, int m, int o){
  
	__shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float Pvalue=0;
	for (int k = 0; k < (m+TILE_WIDTH-1)/(TILE_WIDTH); ++k){
		if (k*TILE_WIDTH + tx < m && row < n){
			Mds[ty][tx] = A[row * m + k*TILE_WIDTH + tx];
		}else{
			Mds[ty][tx] = 0;
		}
		if (k*TILE_WIDTH + ty < m && col < o){
			Nds[ty][tx] = B[(k*TILE_WIDTH + ty) * o + col];
		}else{
			Nds[ty][tx] =0;
		}
		__syncthreads();
		for(int k = 0; k < TILE_WIDTH; ++k){
			Pvalue += Mds[ty][k] * Nds[k][tx];
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

void multMatrixsequential(int *A, int *B, int *C, int n, int m, int o){
	for (int i=0; i<n; i++){
		for (int j=0; j<o; j++){
			int sum=0;
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

int matrixMult( int *A, int *B, int *C,int n, int m ,int o, int flag){
  
	int sizeA=n*m*sizeof(int);
	int sizeB=m*o*sizeof(int);
	int sizeC=n*o*sizeof(int);
  
	int *d_A, *d_B, *d_C;
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

void PrintMatrix(int *matrix, int n, int m){
	printf("\n");
	for(int i=0; i< n; i++){
		for(int j=0 ; j< m;j++){
			printf("%i ",matrix[i*m+j]);
		}
		printf("\n");
	}
}

//////////////////////////////////////////////////////////
//Funcion que  llena las matrices con valores aleatorios//
//////////////////////////////////////////////////////////

void FillMatrix(int *matrix,int n,int m, int r){
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
    
	int *A=(int *) malloc(n*m*sizeof(int));
	
	int *B=(int *) malloc(m*o*sizeof(int));
	
	int *C=(int *) malloc(n*o*sizeof(int));

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