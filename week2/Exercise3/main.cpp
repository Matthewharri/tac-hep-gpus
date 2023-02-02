#include <iostream>

void rps_game(){
    std::cout << "Player 1 enter a choice for Rock (1), Paper (2) or Scissors (3): " << std::endl;
    int player1;
    std::cin >> player1;
    while(player1 != 1 and player1 != 2 and player1 != 3){
        std::cout << "Invalid input, please enter a valid input of 1 (Rock), 2 (Paper) or 3 (Scissors)" << std::endl;
        std::cin >> player1;
    }

    std::cout << "Player 2 enter a choice for Rock (1), Paper (2) or Scissors (3): " << std::endl;
    int player2;
    std::cin >> player2;
    while(player2!=1 and player2!=2 and player2!=3){
        std::cout << "Invalid input, please enter a valid input of 1 (Rock), 2 (Paper) or 3 (Scissors)" << std::endl;
        std::cin >> player2;
    }

    if(player1 == player2){
        std::cout << "It's a tie!" << std::endl;
    }
    else if(player1 == 1 && player2 == 2){
        std::cout << "Player 2 wins!" << std::endl;
    }
    else if(player1 == 1 && player2 == 3){
        std::cout << "Player 1 wins!" << std::endl;
    }
    else if(player1 == 2 && player2 == 1){
        std::cout << "Player 1 wins!" << std::endl;
    }
    else if(player1 == 2 && player2 == 3){
        std::cout << "Player 2 wins!" << std::endl;
    }
    else if(player1 == 3 && player2 == 1){
        std::cout << "Player 2 wins!" << std::endl;
    }
    else if(player1 == 3 && player2 == 2){
        std::cout << "Player 1 wins!" << std::endl;
    }
    else{
        std::cout << "I sure hope this never prints" << std::endl;
    }
}

int main(){
    rps_game();
    return 0;
}


