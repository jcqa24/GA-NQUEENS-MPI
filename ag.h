#include "mpi.h"

int algoritmoGenetico(int N, int p, int np, Chromo *Best, int prob, int numMaxGen, clock_t start, int rank, int size)
{

    int posminlocal;
    int countGen = 0; // Contador de Generaciones
    Chromo *parents = (Chromo *)malloc(sizeof(Chromo) * np);
    Chromo *population = (Chromo *)malloc(sizeof(Chromo) * p);
    reservaMemoria(population, parents, p, np, N);

    int inicio, fin, i;

    int Bestfitness = 100000;

    MPI_Status s;

    inicio = 0;
    fin = p;

    // Generamos la poblacion incial

    InitConf(population, N, inicio, fin); // check

    // Calculamos el fit de la poblacion inicial

    calFit(population, N, inicio, fin); // check

    posminlocal = BuscaMin(population, inicio, fin);

    //gatter
    // MPI_Gather(&posminlocal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    //broadcast Mejor global

    //MPI_Bcast(posminlocal, 1, MPI_INT, 0, MPI_COMM_WORLD);

    copyBest(Best, population[posminlocal], N);
    Bestfitness = population[posminlocal].fitness;

    while ((Bestfitness > 0) && (countGen < numMaxGen))
    {

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

        //critical

    //MPI_Bcast(posminlocal, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (population[posminlocal].fitness < Bestfitness)
        {
            copyBest(Best, population[posminlocal], N);
            Bestfitness = population[posminlocal].fitness;
        }

        //enviar contador y mehor global

        if (rank == 0)
        {

            countGen++;
            //MPI_Bcast(countGen, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    
    return countGen;
    MPI_Finalize();
}
