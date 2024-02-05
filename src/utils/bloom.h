#pragma once

#define FILTER_SIZE 128

//Array of structs
struct BF{
  unsigned long long filter[FILTER_SIZE / 64];

  BF(){
    this->init();
  }
  
  void init(){ //HOST init
    for(int i = 0; i < FILTER_SIZE; i++){
      for(int j = 0; j < FILTER_SIZE / 64; j++){
	
	int offset = i % 64;
	this->filter[j] &= (0ULL << offset); 
      }
    }
  }
};
