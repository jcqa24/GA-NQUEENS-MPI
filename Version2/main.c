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

#include "mpi.h"

int main()
{
    srand(time(NULL));

    int N = 8;          // reinas
    int p = 100;        // poplacion incial
    int np;             // numero de padres
    int prob = 10;      // probabilidad de mutacion
    int numMaxGen = 10; // Numero Maximo de Generaciones
    int countGen = 0;   // contador de Generaciones
    Chromo Best;

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);



    if (rank == 0)
    {
        printf("Agoritmo genetico para N reinias \n");
        printf("Numero de Reinas -> %d\n", N);
        printf("Poblacion inicial -> %d\n", p);
    }

    p = p / size;
    np = p / 2;

    clock_t start = clock();

    countGen = algoritmoGenetico(N, p, np, &Best, prob, numMaxGen, start, rank, size);

    MPI_Finalize();
    return 0;
}
