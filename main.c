#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <omp.h>

#include "definitions.h"
#include "init.h"
#include "mutation.h"
#include "crossover.h"
#include "fitness.h"
#include "selection.h"
#include "messages.h"
#include "ag.h"


int main()
{
    srand(time(NULL));

    int N = 8;             // reinas
    int p = 100;            // poplacion incial
    int np = p / 2;        // numero de padres
    int prob = 10;         // probabilidad de mutacion
    int numMaxGen = 100000; // Numero Maximo de Generaciones
    int countGen = 0; //contador de Generaciones
    Chromo Best;
    Best.fitness = 0;
    Best.config = (int *)malloc(sizeof(int) * N);

    printf("Agoritmo genetico para N reinias \n");
    printf("Numero de Reinas -> %d\n", N);
    printf("Poblacion inicial -> %d\n", p);

    clock_t start = clock();

    countGen = algoritmoGenetico(N, p, np, &Best, prob, numMaxGen, start);

    confFinal(Best, N, start,countGen);

    return 0;
}
