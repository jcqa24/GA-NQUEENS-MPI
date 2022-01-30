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

    int lengths[2] = {1, N};
    MPI_Aint displacements[2];
    MPI_Datatype person_type;

    MPI_Aint base_address;
    MPI_Get_address(&Best, &base_address);
    MPI_Get_address(&Best.fitness, &displacements[0]);
    MPI_Get_address(&Best.config[0], &displacements[1]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);

    MPI_Datatype types[2] = {MPI_INT, MPI_INT};

    MPI_Type_create_struct(2, lengths, displacements, types, &person_type);
    MPI_Type_commit(&person_type);

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

    if (rank == 0)
        confFinal(Best, N, start, countGen);

    MPI_Finalize();
    return 0;
}
