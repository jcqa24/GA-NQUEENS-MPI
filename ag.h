#include "mpi.h"

int algoritmoGenetico(int N, int p, int np, Chromo *Best, int prob, int numMaxGen, clock_t start)
{

    int posminlocal;
    int countGen = 0; // Contador de Generaciones
    Chromo *parents = (Chromo *)malloc(sizeof(Chromo) * np);
    Chromo *population = (Chromo *)malloc(sizeof(Chromo) * p);
    reservaMemoria(population, parents, p, np, N);

    int inicio, fin;

    int Bestfitness = 100000;

    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    inicio = (rank * (p / size));
    fin = inicio + (p / size);
    printf("Soy el hilo %d Incio: %d Fin: %d\n", rank, inicio, fin);

    // Generamos la poblacion incial
    InitConf(population, N, inicio, fin); // check

    // Calculamos el fit de la poblacion inicial
    calFit(population, N, inicio, fin); // check

    posminlocal = BuscaMin(population, inicio, fin);

    //#pragma omp critical

    if (population[posminlocal].fitness < Bestfitness)
    {
        copyBest(Best, population[posminlocal], N);
        Bestfitness = population[posminlocal].fitness;
    }

    while ((Bestfitness > 0) && (countGen < numMaxGen))
    {

        if (rank == 0)
        {
            // Seleccion de padres
            selectChampionship(parents, population, N, p); // check
            // Cruza
            Crossover(parents, population, N, 0, np); // check
        }

        MPI_Barrier(MPI_COMM_WORLD);
        // Mutacion

        mutation(population, prob, N, inicio, fin);

        // Calculo del Fit
        calFit(population, N, inicio, fin);

        // Ordenamos
        // Insertion_sort(population, p);
        posminlocal = BuscaMin(population, inicio, fin);

        //#pragma omp critical

        if (population[posminlocal].fitness < Bestfitness)
        {
            copyBest(Best, population[posminlocal], N);
            Bestfitness = population[posminlocal].fitness;
        }

        if (rank == 0)
        {

            countGen++;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    return countGen;
}
