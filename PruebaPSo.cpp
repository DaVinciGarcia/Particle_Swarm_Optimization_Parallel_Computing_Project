#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <ctime>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


const unsigned int  parts_qty = 1000;
const unsigned int iterations = 1000;
const float min_range_value = -5.12f;
const float max_range_values = 5.12f;
const float w = 0.7f;
const float c1 = 1.5f;
const float c2 = 1.5f;


struct Position {
    float x, y;
    void operator+=(const Position& a) {
        x = x + a.x;
        y = y + a.y;
    }
    void operator=(const Position& a) {
        x = a.x;
        y = a.y;
    }
};

struct Particula {
    Position pBest;
    Position current_position;
    Position velocity;
    float current_value;
};



float randomFloat(float low, float high);
float evalFunct(Position p);
int getTeamBestIndex(Particula* particles, int  parts_qty);
void updateVelocity(Particula &p, Position team_best_position, float w, float c1, float c2);
void updatePosition(Particula &p);


float randomFloat(float low, float high) {
    float range = high-low;
    float pct = static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
    return low + pct * range;
}

float evalFunct(Position p) {
    //return (p.x * p.x - 10 * cosf(2 * M_PI * p.x)) + (p.y * p.y - 10 * cosf(2 * M_PI * p.y));
    return (20 + (p.x * p.x) + (p.y * p.y) - 10*(cosf(2 * M_PI * p.x) + cosf(2 * M_PI * p.y)));
}

int getTeamBestIndex(Particula* particles, int  parts_qty) {
    int best_index = 0;
    float current_team_best = particles[0].current_value;
    for (int i = 1; i <  parts_qty; i++) {
        if (particles[i].current_value < current_team_best) {
            best_index = i;
            current_team_best = particles[i].current_value;
        }
    }
    return best_index;
}


void updateVelocity(Particula &p, Position team_best_position, float w, float c1, float c2) {
    float r_ind = (float)rand() / RAND_MAX;
    float r_team = (float)rand() / RAND_MAX;
    p.velocity.x = w * p.velocity.x +
                   r_ind * c1 * (p.pBest.x - p.current_position.x) +
                   r_team * c2 * (team_best_position.x - p.current_position.x);
    p.velocity.y = w * p.velocity.y +
                   r_ind * c1 * (p.pBest.y - p.current_position.y) +
                   r_team * c2 * (team_best_position.y - p.current_position.y);
}



void updatePosition(Particula &p) {
    p.current_position += p.velocity;
    float newValue = evalFunct(p.current_position);
    if (newValue < p.current_value) {
        p.current_value = newValue;
        p.pBest = p.current_position;
    }
}




int main(void) {
    long start = std::clock();
    std::srand(std::time(NULL));

    Particula* h_particles = new Particula[parts_qty];

    for (int i = 0; i <  parts_qty; i++) {
        h_particles[i].current_position.x = randomFloat(min_range_value, max_range_values);
        h_particles[i].current_position.y = randomFloat(min_range_value, max_range_values);

        h_particles[i].pBest.x = h_particles[i].current_position.x;
        h_particles[i].pBest.y = h_particles[i].current_position.y;

        h_particles[i].current_value = evalFunct(h_particles[i].pBest);

        h_particles[i].velocity.x = randomFloat(min_range_value, max_range_values);
        h_particles[i].velocity.y = randomFloat(min_range_value, max_range_values);
    }

    int team_best_index = getTeamBestIndex(h_particles,  parts_qty);
    Position team_best_position = h_particles[team_best_index].pBest;
    float team_best_value = h_particles[team_best_index].current_value;

    std::cout << "--------------------------------------"  << std::endl;
 
    for (int i = 0; i < iterations; i++) {
        // for each particle
        for (int j = 0; j <  parts_qty; j++) {
            // For each particle calculate velocity
            updateVelocity(h_particles[j], team_best_position, w, c1, c2);
            // Update position and particle best value + position
            updatePosition(h_particles[j]);
        }

        team_best_index = getTeamBestIndex(h_particles,  parts_qty);
        team_best_position = h_particles[team_best_index].pBest;
        team_best_value = h_particles[team_best_index].current_value;
    }


    long stop = std::clock();
    long elapsed = (stop - start) * 1000 / CLOCKS_PER_SEC;


 
    std::cout << "Valor Optimo: " << team_best_value << std::endl;
    //std::cout << "Posicion del Valor: " << team_best_position.toString() << std::endl;
    std::cout << "Tiempo de Ejecucion: " << elapsed << "ms" << std::endl;
    return 0;
}
