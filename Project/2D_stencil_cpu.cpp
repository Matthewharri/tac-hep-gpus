#include <vector>
#include <iostream>
#include <random>
#include <ctime>

#define DSIZE 256
#define RADIUS 3

using Vec2D = std::vector<std::vector<int>>;

Vec2D A(DSIZE, std::vector<int>(DSIZE));
Vec2D B(DSIZE, std::vector<int>(DSIZE));

Vec2D do_2D_stencil(Vec2D matrix){
    Vec2D new_matrix(DSIZE, std::vector<int>(DSIZE));
    for (int i = 0; i < DSIZE; i++){
        for (int j = 0; j < DSIZE; j++){
            if(i < RADIUS || i >= DSIZE - RADIUS || j < RADIUS || j >= DSIZE - RADIUS){
                new_matrix[i][j] = matrix[i][j];
            }
            else{
                for (int k = -RADIUS; k <= RADIUS; k++){
                    new_matrix[i][j] += matrix[i+k][j];
                }
                for (int k = -RADIUS; k <= RADIUS; k++){
                    if(k == 0){
                        continue;
                    }
                    else{
                        new_matrix[i][j] += matrix[i][j+k];
                    }
                }
            }
        }
    }
    return new_matrix; 
}

Vec2D matrix_multiplcation(Vec2D M1, Vec2D M2){
    Vec2D new_matrix(DSIZE, std::vector<int>(DSIZE));
    for (int j = 0; j < DSIZE; j++){
        for (int k = 0; k < DSIZE; k++){
            for (int i = 0; i < DSIZE; i++){
                new_matrix[i][j] += M1[i][k] * M2[k][j];
            }
        }
    }
    return new_matrix;
}

int main() {

    for (int i = 0; i < DSIZE; i++){
        for (int j = 0; j < DSIZE; j++){
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }
    auto new_A = do_2D_stencil(A);
    auto new_B = do_2D_stencil(B);

    auto C = matrix_multiplcation(new_A, new_B);

    return 0;
}
