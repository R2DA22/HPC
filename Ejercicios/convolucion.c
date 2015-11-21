#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cstdlib>


#define TILE_WIDTH 4
#define MAX_MASK_WIDTH 5
__constant__ float s_M[MAX_MASK_WIDTH];
/////////////////////////////////////
//Funcion que  imprime las matrices//
/////////////////////////////////////

void PrintMatrix(float*matrix, int n, int m){
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

///////////////////////////////////////
//Funcion Convolucion BASICA PARALELA//
///////////////////////////////////////


__global__ void convolucionBasicKernel(float *N, float *M, float *P,
 int Mask_Width, int Width)
 {
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   float Pvalue = 0;
   int N_start_point = i - (Mask_Width/2);
   
   for (int j = 0; j < Mask_Width; j++)
   {
     if (N_start_point + j >= 0 && N_start_point + j < Width)
     {
       Pvalue += N[N_start_point + j] * M[j];
       
			 
     }
     
   }
   P[i] = Pvalue;
}
////////////////////////////////////////
//Funcion Convolucion PARALELA CACHING//
////////////////////////////////////////

__global__ void convolucionCachingKernel(float *N,float *P,int Mask_Width, int Width){

   int i = blockIdx.x*blockDim.x + threadIdx.x;
   float Pvalue = 0;
   int N_start_point = i - (Mask_Width/2);
   
   for (int j = 0; j < Mask_Width; j++)
   {
     if (N_start_point + j >= 0 && N_start_point + j < Width)
     {
       Pvalue += N[N_start_point + j] * s_M[j];
       
			 
     }
     
   }
   P[i] = Pvalue;
}

///////////////////////////////////////
//Funcion Convolucion PARALELA TILING//
///////////////////////////////////////

__global__ void convolucionTiledKernel(float *N,float *P, int Mask_Width, int Width){

  int i = blockIdx.x*blockDim.x+threadIdx.x;
  
  __shared__ float Nds[TILE_WIDTH + MAX_MASK_WIDTH-1];
  
  int n =Mask_Width/2;

  int halo_index_left = (blockIdx.x-1)*blockDim.x+threadIdx.x;
  if(threadIdx.x >= blockDim.x-n){
    Nds[threadIdx.x-(blockDim.x-n)] = (halo_index_left < 0)? 0 : N[halo_index_left];
  }

  Nds[n+threadIdx.x] = N[blockIdx.x*blockDim.x+threadIdx.x];

  int halo_index_right = (blockIdx.x+1)*blockDim.x+threadIdx.x;
  if(threadIdx.x < n){
    Nds[n+blockDim.x+threadIdx.x]=(halo_index_right >= Width) ? 0 : N[halo_index_right];
  }

  __syncthreads();

  float Pvalue = 0;
  for(int j=0; j<Mask_Width; j++){
    Pvalue += Nds[threadIdx.x+j]*s_M[j];
  }
  P[i]=Pvalue;
}

/////////////////////////////////////////
//Funcion Convolucion BASICA SECUENCIAL//
/////////////////////////////////////////

void convolucionBasicSecuencial(float *N, float *M, float *P,int Mask_Width, int Width){
   

	float Pvalue = 0;
	
	for (int i = 0; i < Width; i++){
    
    int N_start_point = i - (Mask_Width/2);
		for (int j = 0; j < Mask_Width; j++){
			
     		if (N_start_point + j >= 0 && N_start_point + j < Width){
       				Pvalue += N[N_start_point + j] * M[j];
          		
     		}
     
		}
		P[i] = Pvalue;
    Pvalue=0;
	}
}
///////////////////////////////////
//Funcion que tiene codigo cuda c//
///////////////////////////////////

float Convolucion( float *N, float *M, float *P,int Mask_High,int Mask_Width, int n,int m, int flag){
  	float T=0;
	int sizeN=n*m*sizeof(int);
	int sizeP=n*m*sizeof(int);
	float *d_N, *d_P;
  	if(flag==0){
		int sizeM=Mask_High*Mask_Width*sizeof(int);

		float *d_M;
		//Reservo Memoria en el dispositivo
		cudaMalloc((void **)&d_M,	sizeM);
		cudaMalloc((void **)&d_N, sizeN);
		cudaMalloc((void **)&d_P,	sizeP);
		
		clock_t t;
		t=clock();
		//Copio los datos al device
		cudaMemcpy(d_N, N, sizeN, cudaMemcpyHostToDevice);
		cudaMemcpy(d_M, M, sizeM, cudaMemcpyHostToDevice);
	  
		dim3 dimBlock(32.0,32.0); //mayor cantidad de hilos por bloque
		dim3 dimGrid(ceil((float)n/dimBlock.x),ceil((float)n/dimBlock.y));
		// Ejecuto el Kernel (del dispositivo)
	
		convolucionBasicKernel<<<dimGrid,dimBlock>>>(d_N, d_M, d_P,Mask_Width,m);
    	cudaMemcpy(P, d_P, sizeP, cudaMemcpyDeviceToHost);
		printf("Convolucion Basica\t\t\t\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
		T=(clock()-t)/(double)CLOCKS_PER_SEC;
      cudaFree(d_M);
	}
	if(flag==1){
		
		//Reservo Memoria en el dispositivo
		cudaMalloc((void **)&d_N, sizeN);
		cudaMalloc((void **)&d_P,	sizeP);
		
		clock_t t;
		t=clock();
		//Copio los datos al device
		cudaMemcpy(d_N, N, sizeN, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(s_M, M, Mask_Width*sizeof(float));
	  
		dim3 dimBlock(32.0,32.0); //mayor cantidad de hilos por bloque
		dim3 dimGrid(ceil((float)n/dimBlock.x),ceil((float)n/dimBlock.y));
		// Ejecuto el Kernel (del dispositivo)
		convolucionCachingKernel<<<dimGrid,dimBlock>>>(d_N,d_P,Mask_Width,m);
		cudaDeviceSynchronize();
    	cudaMemcpy(P, d_P, sizeP, cudaMemcpyDeviceToHost);
		printf("Convolucion Catching\t\t\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
		T=(clock()-t)/(double)CLOCKS_PER_SEC;
	}
	if(flag==2){
		//Reservo Memoria en el dispositivo
		cudaMalloc((void **)&d_N, sizeN);
		cudaMalloc((void **)&d_P,	sizeP);
		
		clock_t t;
		t=clock();
		//Copio los datos al device
		cudaMemcpy(d_N, N, sizeN, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(s_M, M, Mask_Width*sizeof(float));
	  
		dim3 dimBlock(5.0,5.0); //mayor cantidad de hilos por bloque
		dim3 dimGrid(ceil((float)n/dimBlock.x),ceil((float)n/dimBlock.y));
		// Ejecuto el Kernel (del dispositivo)
		convolucionTiledKernel<<<dimGrid,dimBlock>>>(d_N,d_P,Mask_Width,m);
    	cudaMemcpy(P, d_P, sizeP, cudaMemcpyDeviceToHost);
		printf("Convolucion Tiling\t\t\t\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
		T=(clock()-t)/(double)CLOCKS_PER_SEC;
	}
	cudaFree(d_P);
	return T;
}

//////////////////////////
//Funcion main principal//
//////////////////////////

int main(){
	int n=1;
	int  m1[] = {512,1024,5000,10000,20000,40000,80000,160000,320000,500000};
	float TS=0,TPB=0 ,TPC=0,TPT=0;
    int Mask_High=1;
    int Mask_Width=MAX_MASK_WIDTH;
    for (int i = 0; i < 10; ++i){
    	printf("==============Tiempos de ejecucion %i ==============\n",i+1);	
    	int m=m1[i];
   
    		
		float *M=(float *) malloc(Mask_High*Mask_Width*sizeof(float));
			
		float *N=(float *) malloc(n*m*sizeof(float));
		
		float *P=(float *) malloc(n*m*sizeof(float));
			
		FillMatrix(M, Mask_High, Mask_Width,10);
		FillMatrix(N, n, m,10);
		printf("Secuencial");
		for (int j = 0; j < 6; ++j){
			clock_t t;
			t=clock();
			convolucionBasicSecuencial(N,M,P,Mask_Width,m);
			
			printf("Convolucion secuencial\t\t\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
			TS+=(clock()-t)/(double)CLOCKS_PER_SEC;
		}
		printf("Promedio de tiempo: %.8f\n",TS/6);
		printf("Basica");
		for (int i = 0; i < 6; ++i)
		{
			TPB+=Convolucion(N,M,P,Mask_High,Mask_Width,n,m,0);
			
		}
		printf("Tiempo promedio: %.8f\n",TPB/6);
		printf("Caching");
		for (int i = 0; i < 6; ++i)
		{
			
			TPC+=Convolucion(N,M,P,Mask_High,Mask_Width,n,m,1);
	
		}
		printf("Tiempo promedio: %.8f\n",TPC/6);
		printf("Tiling");
		for (int i = 0; i < 6; ++i)
		{
			
			TPT+=Convolucion(N,M,P,Mask_High,Mask_Width,n,m,2);
		}
		printf("Tiempo promedio: %.8f\n",TPT/6);
		/*PrintMatrix(M,Mask_High,Mask_Width);
		PrintMatrix(N,n,m);
		PrintMatrix(P,n,m);*/
		free(M);
		free(N);
		free(P);
	}
	
	
	return 0;

}