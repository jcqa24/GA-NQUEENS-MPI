void ip(Chromo *population, int p, int N)
{
    for (int j = 0; j < p; j++)
    {
        printf("Gen: %d [ ", j);
        for (int i = 0; i < N; i++)
        {
            printf(" %d ", population[j].config[i] + 1);
        }
        printf("] fitnes ==> %d \n", population[j].fitness);
    }
}

void printConf(int *conf, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (conf[j] == i)
            {
                printf(" %c ", 'X');
            }
            else
            {
                printf(" %c ", '_');
            }
        }
        printf("\n");
    }
}

void confFinal(Chromo Best, int N, clock_t start, int numMaxGen)
{
    printf("\n\n=============================================\n");
    printf("La mejor solucion tiene un fit de %d \n", Best.fitness);
    printf("La mejor solucion es: [ ");
    for (int i = 0; i < N; i++)
    {
        printf(" %d ", Best.config[i] + 1);
    }
    printf("]\n");

    if (N > 15)
    {
        printConf(Best.config, N);
    }
    else
    {
        imprimeTablero(Best.config, N);
    }

    printf("\n===============================================\n");

    printf("Numero de generaciones transcurridas: %d\n", numMaxGen);
    printf("Tiempo transcurrido: %2.10f seg.\n", ((double)clock() - start) / CLOCKS_PER_SEC);
}

void imprimeTablero(int *conf, int N)
{
    printf("\n\n");
    int i, j, aux;
    aux = N;
    printf("  ");
    for (i = (N - 1); i >= 0; i--)
    {
        printf("   %c   ", i + 65);
    }
    printf("\n  ");
    for (i = 0; i < N; i++)
    {
        printf(" ------");
    }
    printf("\n");
    for (i = 0; i < N; i++)
    {
        printf("%d |", aux--);
        for (j = 0; j < N; j++)
        {
            printf("      |");
        }
        printf("\n  |");
        for (j = 0; j < N; j++)
        {
            if (conf[j] == i)
            {
                printf("   Q  |");
            }
            else
            {
                printf("      |");
            }
        }
        printf("\n");
        printf("  ");
        for (int k = 0; k < N; k++)
        {
            printf(" ------");
        }
        printf("\n");
    }
    printf("\n\n");
}