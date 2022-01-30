#include "mpi.h"

int algoritmoGenetico(int N, int p, int np, Chromo *Best, int prob, int numMaxGen, clock_t start, int rank, int size)
{

    int posminlocal, idbestglobal=0;
    int countGen = 0; // Contador de Generaciones
    Chromo *parents = (Chromo *)malloc(sizeof(Chromo) * np);
    Chromo *population = (Chromo *)malloc(sizeof(Chromo) * p);

    reservaMemoria(population, parents, p, np, N);

    int inicio, fin, i;

    int Bestfitness = 100000;

    // printf("Cada hilo trabajara: %d \n",p);

    MPI_Status s;

    inicio = 0;
    fin = p;

    int *CandidateBest = (int *)malloc(sizeof(int) * size);

    // Generamos la poblacion incial

    InitConf(population, N, inicio, fin); // check

    // Calculamos el fit de la poblacion inicial

    calFit(population, N, inicio, fin); // check

    posminlocal = BuscaMin(population, inicio, fin);

    

    MPI_Gather(&population[posminlocal].fitness, 1, MPI_INT, &CandidateBest[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        idbestglobal = 0;

        for (i = 1; i < size; i++)
        {
            printf("El proceso minimo es: %d  \t i::>%d\n",idbestglobal,i);
            if (Bestfitness > CandidateBest[i])
            {
                Bestfitness = CandidateBest[i];
                idbestglobal = i;
            }
        }
    }

    printf("inicialdo BCast....\n");
    MPI_Bcast(&idbestglobal, 1, MPI_INT, 0, MPI_COMM_WORLD);

   
    printf("El mejor esta en %d \n",idbestglobal);
    
    MPI_Bcast(&population[posminlocal], sizeof(Best), MPI_BYTE, idbestglobal, MPI_COMM_WORLD);
    MPI_Bcast(&Bestfitness, 1, MPI_INT, 0, MPI_COMM_WORLD);
    

    printf("Saliedno Bcast %d\n ",rank);
    
    

    

    while ((Bestfitness > 0) && (countGen < numMaxGen))
    {
    
        printf("\t\t\tEntrando en while ... con mejor fit %d\n",Bestfitness);

        // Seleccion de padres

        selectChampionship(parents, population, N, p); // check

        // Cruza

        Crossover(parents, population, N, 0, np); // check

        // Mutacion

        mutation(population, prob, N, inicio, fin);

        // Calculo del Fit
        calFit(population, N, inicio, fin);

        // Ordenamos
        // Insertion_sort(population, p);
        posminlocal = BuscaMin(population, inicio, fin);

        // critical

        MPI_Gather(&population[posminlocal].fitness, 1, MPI_INT, &CandidateBest[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        idbestglobal = 0;

        for (i = 1; i < size; i++)
        {
            printf("El proceso minimo es: %d  \t i::>%d\n",idbestglobal,i);
            if (Bestfitness > CandidateBest[i])
            {
                Bestfitness = CandidateBest[i];
                idbestglobal = i;
            }
        }
    }

    printf("inicialdo BCast....\n");
    MPI_Bcast(&idbestglobal, 1, MPI_INT, 0, MPI_COMM_WORLD);

   
    printf("El mejor esta en %d \n",idbestglobal);
    
    MPI_Bcast(&population[posminlocal], sizeof(Chromo), MPI_BYTE, idbestglobal, MPI_COMM_WORLD);
    MPI_Bcast(&Bestfitness, 1, MPI_INT, 0, MPI_COMM_WORLD);
    

     printf("Saliedno Bcast %d\n ",rank);

        if (rank == 0)
        {

            countGen++;
        }

        MPI_Bcast(&countGen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    return countGen;
}
