#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>
#include <map>
#include <math.h>
#include <utility>
#include <cmath>

#define dimensiones 4
#define cluster 2

using namespace std;

void Fillcentro(double *matrix,int n,int m, int r);
void FillMatrix(int *matrix,int n,int m, int r);
void PrintCentros(double * centros, int n, int m);
void PrintMatrix(int *matrix, int n, int m);
double distancia(const double *centro,int *nodo, int n,int m);

class asignacion{

	private:
		int * v;
		int id_centro;
		double dist;
	public:
		asignacion(){}
		asignacion(int *x,int id, const double *centro,int n, int m){
			v=x;
			id_centro=id;
      
			dist=distancia(centro,x,n,m);
		}
		
		int *Getv()const{
			return v;
		}
    
		int Getid()const{
			return id_centro;
		}
		double Getdist()const{
			return dist;
		}

};
__global__ void newCenters(int* A, int* out,double* C, int n, int m, int k){
  
  int tid= blockDim.x*blockIdx.x+threadIdx.x;
  for(int j=0; j< k ; j++){
  	if(tid < n){
      double sum=0;
      if(out[tid] == j){
        
        for(int i=0; i < m;i++){
          //printf("(%.1f + %i)",C[i+j*m],A[i+tid*m]);
          sum+=A[i+tid*m];
          
        }
        
      }
      C[tid+j*m]=sum;
    }
  }
  
}

///////////////////////////////////////////////
//Funcion Kmeans asignacion paralela en cuda //
///////////////////////////////////////////////

__global__ void kmeansParallel(int* A, double* C, int* out, int n, int m, int k){
	 
  int flag=0;
  int iter=0;
  double aux=0;
  double temp=0;
  int id_centro=0;
	int tid= blockDim.x*blockIdx.x+threadIdx.x;
  
  __shared__ double dc_shared[cluster*dimensiones];
  
	int threadID = threadIdx.x;
  
 	while(threadID < k*m){
        dc_shared[threadID] = C[threadID];

        threadID += blockDim.x;
    }
  __syncthreads();
 
	if ((tid < n)){
      
			for(int centerIdx=0; centerIdx < k; centerIdx++){
				  
          aux=0;
					for (int dimIdx = 0; dimIdx < m; dimIdx++){
              //printf("(%i -%.1f)",A[tid * m + dimIdx] , C[centerIdx * m + dimIdx]);
							aux+=(A[tid * m + dimIdx] - dc_shared[centerIdx * m + dimIdx])*(A[tid * m + dimIdx] - dc_shared[centerIdx * m + dimIdx]);
							
					}
          //printf("%.1f ",aux);
					aux=sqrtf(aux);
           
					if(flag == 0){
						temp=aux;
				    flag=1;
            
					}
					if(aux < temp){
						temp=aux;
						id_centro=iter;  
					}
          iter++;
          //printf("%.1f ",aux); 
      }
       //printf("id=%i ",id_centro);
       out[tid]=id_centro;
       temp=0;
			 flag=0;
			 id_centro=0;
       iter=0;		
	}
	__syncthreads();
}


///////////////////////////////////
//Funcion que tiene codigo cuda c//
///////////////////////////////////

int kmeans( int *A,double *Centros,int* out,int n, int m,int k){
  double *NC=(double *) malloc(k*m*sizeof(double));
  Fillcentro(NC,k,m,1);
	int sizeA=n*m*sizeof(int);
  int sizeO=n*sizeof(int);
	int sizeC=k*m*sizeof(double);
  int sizeCN=k*m*sizeof(double);
	int *d_A;
  double *d_C;
  double *d_NC;
  int *d_O;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A,	sizeA);
	cudaMalloc((void **)&d_C,	sizeC);
  cudaMalloc((void **)&d_NC,sizeCN);
	cudaMalloc((void **)&d_O,	sizeO);
	//clock_t t;
	//t=clock();
	//Copio los datos al device
	cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, Centros, sizeC, cudaMemcpyHostToDevice);
  cudaMemcpy(d_NC, NC, sizeCN, cudaMemcpyHostToDevice);
	dim3 dimBlock(32,1); //mayor cantidad de hilos por bloque
	dim3 dimGrid(ceil((float)n/dimBlock.x),ceil((float)n/dimBlock.y));
	// Ejecuto el Kernel (del dispositivo)
	
	kmeansParallel<<<dimGrid,dimBlock>>>(d_A,d_C,d_O,n,m,k);
  cudaMemcpy(out, d_O, sizeO, cudaMemcpyDeviceToHost);
	//printf("kmeans paralela \t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
	PrintMatrix(out,n,1);
	newCenters<<<dimGrid,dimBlock>>>(d_A,d_O,d_NC,n,m,k);
  cudaMemcpy(Centros, d_NC, sizeC, cudaMemcpyDeviceToHost);
	PrintCentros(Centros,k,m);
	cudaFree(d_A);
	cudaFree(d_C);
	return 0;
}

/////////////////////////////////////
//Funcion que  imprime las matrices//
/////////////////////////////////////

void PrintMatrix(int *matrix, int n, int m){
	cout<<endl;
	for(int i=0; i< n; i++){
		for(int j=0 ; j< m;j++){
			cout<<" "<<matrix[i*m+j];
		}
		cout<<endl;
	}
}

void PrintCentros(double * centros, int n, int m){
	cout<<endl;
	for(int i=0; i< n; i++){
		for(int j=0 ; j< m;j++){
			cout<<" "<<centros[i*m+j];
		}
		cout<<endl;
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
void Fillcentro(double *matrix,int n,int m, int r){
	for(int i=0; i < n*m; i++){
		matrix[i]=rand()% r;
	}
}
////////////////////////////
//Funcion   obtiener datos//
////////////////////////////

void GetCentros(int *matrix,double *centros,int m,int cant,int K){
  int n=0;
  int auxIt=0;
  int temporal=-1;
  int var=-1;
  for(int j=0; j < K; j++){
    while(temporal==var)
      temporal=rand() % cant;
    
    var=temporal;
    n=temporal*m; 
    for(int i=0; i < m; i++){
      centros[auxIt]=matrix[n];
      n++;
      auxIt++;
    }
  }
}

int * Getnodo(int *matrix,int m,int iter){
  int n=iter*m;
  int *nodo=(int *) malloc(m*sizeof(int));
  for(int i=0; i < m; i++){
      nodo[i]=matrix[n];
      n++;
  }
  return nodo;
}
double* Getcentro(double *centros,int m,int iter){
  int n=iter*m; 
  double *centro=(double *) malloc(m*sizeof(double));
  for(int i=0; i < m; i++){
      centro[i]=centros[n];
      n++;
  }
  return centro;
}
////////////////////////////////////////////
//Funcion para medir distancias Euclediana//
///////////////////////////////////////////

double distancia(const double *centro,int *nodo, int n,int m){
  double aux=0;
  int i=0;
  for (i = 0; i < m; i++){
	   aux+=pow((centro[i]-nodo[i]),2);
  }
  return aux=sqrt(aux);
}

pair<double,double*> asignar(int *matriz, double *centros, map<int, asignacion> &tablaAsignaciones,int K , int m , int n){
  double aux=0;
  double temp=0;
  int iter=0;
  int flag=0;
  int id_centro=0;
  int *nodo=(int *) malloc(m*sizeof(int));
  double *centro=(double *) malloc(m*sizeof(double));
  pair<double,double*> result;
  for(int i=0; i < n; i++ ){
    nodo=Getnodo(matriz,m,i);
    for(int j=0; j < K; j++){
     		centro=Getcentro(centros,m,j);
	    	aux=distancia(centro,nodo,n,m);    
	    	if(flag == 0){
	    		temp=aux;
          flag=1;
	    	}
	    	if(aux <= temp){
	    		temp=aux;
	    		id_centro=iter;  
	    	}
        //cout<<aux<<" ";
        iter++;
        
    }
    centro=Getcentro(centros,m,id_centro);

    asignacion asig(nodo,id_centro,centro,n,m);
    tablaAsignaciones[i]=asig;
    //cout<<"id="<<id_centro<<endl;
    iter=0;
    temp=0;
    flag=0;
    id_centro=0;
  }
  //PrintMatrix(tablaAsignaciones[2].Getv(),1,m);
  //cout << tablaAsignaciones[2].Getdist()<<endl;
  double *centros_new=(double *) malloc(K*m*sizeof(double));
  Fillcentro(centros_new, K, m,1);
  //PrintCentros(centros,K,m);
  //PrintCentros(centros_new,K,m);
  int *aux2=(int *)malloc(K*sizeof(int));
  for (int j=0; j < K ; j++){
  	    
  	    int S=0;
		for (map<int,asignacion>::iterator it=tablaAsignaciones.begin(); it!=tablaAsignaciones.end(); it++){
			int* v_aux =it->second.Getv();
			//PrintMatrix(v_aux,1,m);
			int itw=j*m;

			id_centro=it->second.Getid();
			if(id_centro == j){
				for (int i = 0; i < m; i++){
					   //cout << centros_new[itw]<<"+"<<v_aux[i]<<endl;
						centros_new[itw]=centros_new[itw]+v_aux[i];
						itw++;


				}
                //cout << "hola"<<endl;
				S++;
                
			}	
			
        }
        aux2[j]=S;	
	    //PrintCentros(centros_new,1,m);
  }
  double sum_dist=0;
  //PrintCentros(centros_new,K,m);
  for (int x = 0; x < K; x++){
  	        //cout<<aux2[x]<<" ";
        	int itr=x*m;
			
	        for (int i = 0; i < m; i++){
					if(aux2[x]!=0){
						centros_new[itr]=((double)centros_new[itr]/aux2[x]);
					}else{
						centros_new[itr]=0;
					}
					if(x==0)
						sum_dist+=tablaAsignaciones[i].Getdist();
					itr++;
		    }
  }
  //cout<<sum_dist<<endl;
  //PrintCentros(centros_new,K,m);
  result.first=sum_dist;
  result.second=centros_new;
  free(nodo);
  free(centro);
  //free(centros_new);
  //free(aux2);
  return  result;
}

//////////////////////////
//Funcion main principal//
//////////////////////////

int main(){
	int n=4089;
	int m=4089;
    int K=2;
    double error=0.0;
    double alpha=1;
    srand (time(NULL));
 	int *A=(int *) malloc(n*m*sizeof(int));
	int *out=(int *) malloc(n*sizeof(int));
	double *centros=(double *) malloc(K*m*sizeof(double));
    map<int, asignacion> tablaAsignaciones;
    pair<double,double*> result;
  
  
	FillMatrix(A, n, m,4089);
	//PrintMatrix(A,n,m);
	
    GetCentros(A,centros,m,n,K);
    //PrintCentros(centros,K,m);
  clock_t t;
	t=clock();
  int art=0;
    while(art < 20){
    	
    	result=asignar(A,centros,tablaAsignaciones,K,m,n);
      centros=result.second;
      //PrintCentros(centros,K,m);
    	error=abs(error-result.first);
    	cout <<"error: "<< error <<endl;
		  if (error < alpha){
        //centros=result.second;
        //PrintCentros(centros,K,m);
        break;
			}
			
      art++;
    }
 cout<<"Tiempo: "<<(clock()-t)/(double)CLOCKS_PER_SEC<<endl;
	//kmeans(A,centros,out,n,m,K);
	free(A);
	free(centros);
	
	return 0;

}

