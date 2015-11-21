#include "highgui.h"
#include "cv.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define Mask_size  3
#define TILE_SIZE  32

using namespace cv;


__constant__ char M[Mask_size*Mask_size];

__device__ unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return  value;
}



///////////////////////////////////////
//Funcion Convolucion Memoria Global///
///////////////////////////////////////

__global__ void MemoriaGlobal(unsigned char *In,char *Mask, unsigned char *Out,int Mask_Width,int Rowimg,int Colimg)
{

   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int Pvalue = 0;

   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++)
   {
       for(int j = 0; j < Mask_Width; j++ )
       {
        if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)
        &&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
        {
          Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * Mask[i*Mask_Width+j];
        }
       }
   }

   Out[row*Rowimg+col] = clamp(Pvalue);

}

////////////////////////////////////////
//Funcion Convolucion  Memoria Constante/
////////////////////////////////////////

__global__ void MemoriaConstante(unsigned char *In,unsigned char *Out,int Mask_Width,int Rowimg,int Colimg)
 {
   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int Pvalue = 0;

   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++)
   {
       for(int j = 0; j < Mask_Width; j++ )
       {
         if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)
         &&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
         {
           Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * M[i*Mask_Width+j];
         }
       }
    }

   Out[row*Rowimg+col] = clamp(Pvalue);
}

///////////////////////////////////////
//Funcion Convolucion  Memoria Compartida//
///////////////////////////////////////


__global__ void MemoriaCompartida(unsigned char *imageInput,unsigned char *imageOutput,
 int maskWidth, int width, int height)
{
    __shared__ float N_ds[TILE_SIZE + Mask_size - 1][TILE_SIZE + Mask_size - 1];
    int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+Mask_size-1), destX = dest % (TILE_SIZE+Mask_size-1),
        srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
        src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + Mask_size - 1), destX = dest % (TILE_SIZE + Mask_size - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    if (destY < TILE_SIZE + Mask_size - 1)
    {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    int Pvalue = 0;
    int y, x;
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            Pvalue += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskWidth + x];
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (y < height && x < width)
        imageOutput[(y * width + x)] = clamp(Pvalue);
    __syncthreads();
}




float convolucion2D(Mat imagen,unsigned char *imagenInicial,unsigned char *imagenResultado,char *h_Mask,int Mask_Width,int Row,int Col, int flag){
  float T=0;
  int Size_of_bytes = sizeof(unsigned char)*Row*Col*imagen.channels();
  int Mask_size_bytes = sizeof(char)*(Mask_size*Mask_size);
  unsigned char *d_A, *d_B;
  char *d_Mask;


 
  cudaMalloc((void**)&d_A,Size_of_bytes);
  cudaMalloc((void**)&d_B,Size_of_bytes);
  cudaMalloc((void**)&d_Mask,Mask_size_bytes);

  if(flag==0){
    clock_t t;
    t=clock();

    cudaMemcpy(d_A,imagenInicial,Size_of_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mask,h_Mask,Mask_size_bytes,cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(s_M,h_Mask,Mask_size_bytes);

    dim3 dimBlock(32,32,1); //mayor cantidad de hilos por bloque
    dim3 dimGrid(ceil(Row/dimBlock.x),ceil(Col/dimBlock.y));
    MemoriaGlobal<<<dimGrid,dimBlock>>>(d_A,d_Mask,d_B,Mask_Width,Row,Col);
    cudaMemcpy (imagenResultado,d_B,Size_of_bytes,cudaMemcpyDeviceToHost);
    //printf("Convolucion Basica\t\t\t\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
    T=(clock()-t)/(double)CLOCKS_PER_SEC;
    cudaFree(d_Mask);
  } 

  if(flag==1){
    clock_t t;
    t=clock();

    cudaMemcpy(d_A,imagenInicial,Size_of_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mask,h_Mask,Mask_size_bytes,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M,h_Mask,Mask_size_bytes);
    dim3 dimBlock(32,32); //mayor cantidad de hilos por bloque
    dim3 dimGrid(ceil(Row/dimBlock.x),ceil(Col/dimBlock.y));
    
    MemoriaConstante<<<dimGrid,dimBlock>>>(d_A,d_B,Mask_Width,Row,Col);
    cudaDeviceSynchronize();
    cudaMemcpy (imagenResultado,d_B,Size_of_bytes,cudaMemcpyDeviceToHost);
    //printf("Convolucion caching\t\t\t\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
    T=(clock()-t)/(double)CLOCKS_PER_SEC;
    
  }
  if(flag==2){
    clock_t t;
    t=clock();

    cudaMemcpy(d_A,imagenInicial,Size_of_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mask,h_Mask,Mask_size_bytes,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M,h_Mask,Mask_size_bytes);
    dim3 dimBlock(32,32); //mayor cantidad de hilos por bloque
    dim3 dimGrid(ceil(Row/dimBlock.x),ceil(Col/dimBlock.y));
    MemoriaCompartida<<<dimGrid,dimBlock>>>(d_A,d_B,Mask_Width,Row,Col);
    cudaMemcpy (imagenResultado,d_B,Size_of_bytes,cudaMemcpyDeviceToHost);
    //printf("Convolucion Tiling\t\t\t\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
    T=(clock()-t)/(double)CLOCKS_PER_SEC;
    

  }
  
  cudaFree(d_A);
  cudaFree(d_B);
  return T;
}

int main( ){
  
  Mat imagen;

  float tiempo=0;
  float promedio=0;
  
  imagen = imread( "./inputs/img1.jpg",0);
  int Mask_Width =  Mask_size;
  Size dimensiones = imagen.size();
  int Row = dimensiones.width;
  int Col = dimensiones.height;
  char h_Mask[] = {-1,0,1,-2,0,2,-1,0,1};
  
  if( !imagen.data ){
    return -1;
  }

  unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*imagen.channels());
  unsigned char *imgOut = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*imagen.channels());

  img = imagen.data;
  
  for (int j = 0; j < 20; ++j ){
      
      tiempo+=convolucion2D(imagen,img,imgOut,h_Mask,Mask_Width,Row,Col,0);
  }
  promedio=tiempo/20;
  printf("%.8f\n",promedio);
  
  

  Mat gray_image;
  gray_image.create(Col,Row,CV_8UC1);
  gray_image.data = imgOut;
  imwrite("./outputs/1112905491.png",gray_image);
  return 0;
}