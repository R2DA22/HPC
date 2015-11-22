#include "highgui.h"
#include "cv.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace cv;

int main( ){
  
  Mat imagen, imagenEscalaGrises;
  Mat imagenResultado;

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  float promedio=0;
  float tiempo=0;
  
  imagen = imread( "./inputs/img4.jpg" );
  
  if(!imagen.data )
    return -1;
  
  
    
  for(int i =0; i<20; i++){
      clock_t t;
      t=clock();

      cvtColor( imagen, imagenEscalaGrises, CV_RGB2GRAY );

      Sobel( imagenEscalaGrises, grad_x, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT );
      convertScaleAbs( grad_x, abs_grad_x );


      Sobel( imagenEscalaGrises, grad_y, CV_8UC1, 0, 1, 3, 1, 0, BORDER_DEFAULT );
      convertScaleAbs( grad_y, abs_grad_y );

      addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imagenResultado );
      tiempo+=(clock()-t)/(double)CLOCKS_PER_SEC;
  }
  promedio=tiempo/20;
  
  printf("promedio %.8f\n",promedio);
  tiempo=0;
 	
  
  imwrite("./outputs/1112905491.png",imagenResultado);
  return 0;
}