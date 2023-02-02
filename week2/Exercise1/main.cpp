#include <iostream>

void swap_function(int a[], int b[], int size){
    for(int i = 0; i < size; i++){
        auto tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}


int main(){

    int a[10] = {0,1,2,3,4,5,6,7,8,9};
    int b[10] = {9,8,7,6,5,4,3,2,1,0};
    std::cout << "a before swapping: " << std::endl; 
    for(int i = 0; i < 10; i++)
        std::cout << a[i] << " ";
    std::cout << std::endl;
    std::cout << "b before swapping: " << std::endl; 
    for(int i = 0; i < 10; i++)
        std::cout << b[i] << " ";
    std::cout << std::endl;

    swap_function(a,b,10);

    std::cout << "a after swapping: " << std::endl;
    for(int i = 0; i < 10; i++)
        std::cout << a[i] << " ";
    std::cout << std::endl;
    std::cout << "b after swapping: " << std::endl;
    for(int i = 0; i < 10; i++)
        std::cout << b[i] << " ";
    std::cout << std::endl;

    return 0;
}