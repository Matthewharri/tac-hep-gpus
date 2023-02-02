#include <iostream>
#include <string>

struct student{
    const std::string name;
    const std::string email;
    const std::string username;
    const std::string experiment;

};

void print_info(student s[], int size){
    for(int i = 0; i<size; i++){
        std::cout << "Name of student: " << s[i].name << ", ";
        std::cout << " Email of student: " << s[i].email << ", ";
        std::cout << " Username of student: " << s[i].username << ", ";
        std::cout << " Experiment of student: " << s[i].experiment << std::endl;

    }
}

int main(){
    std::string names[6] = {
        "Matt",
        "Elise",
        "Ryan",
        "Shivani",
        "Stephanie",
        "Trevor"
    };

    std::string emails[6] = {
        "Matthewharri@umass.edu",
        "emchavez@wisc.edu",
        "rsimeon@wisc.edu",
        "lomte@wisc.edu",
        "skwan@princeton.edu",
        "twnelson2@wisc.edu"
    };

    std::string usernames[6] = {
        "Matthewharri",
        "emchavez",
        "rsimeon",
        "lomte",
        "skkwan",
        "twnelson2"
    };

    std::string experiments[6] = {
        "ATLAS",
        "CMS",
        "CMS",
        "CMS",
        "CMS",
        "CMS"
    };

    student students[6] = {
        {names[0], emails[0], usernames[0], experiments[0]},
        {names[1], emails[1], usernames[1], experiments[1]},
        {names[2], emails[2], usernames[2], experiments[2]},
        {names[3], emails[3], usernames[3], experiments[3]},
        {names[4], emails[4], usernames[4], experiments[4]},
        {names[5], emails[5], usernames[5], experiments[5]}
    };

    print_info(students, 6);


    return 0;
}