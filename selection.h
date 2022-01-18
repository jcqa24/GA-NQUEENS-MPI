void Insertion_sort(Chromo *population, int p)
{

    int i, j;
    Chromo temp;

    for (i = 1; i < p; i++)
    {
        temp = population[i];
        for (j = i; j > 0 && population[j - 1].fitness > temp.fitness; j--)
        {
            population[j] = population[j - 1];
        }
        population[j] = temp;
    }
}



void shuffle(Chromo *population, int p, int N)
{
    int i, j;
    Chromo temp;
    temp.config = (int *)malloc(sizeof(int) * N);
    int c = 0;                     // contador de cambios;
    int nc = rand() % (p - 2) + 2; // numero de cambios
    while (c <= nc)
    {

        i = rand() % (p);
        j = rand() % (p);
        if (j != i)
        {
            temp = population[i];
            population[i] = population[j];
            population[j] = temp;
            c++;
        }
    }
}

int BuscaMin(Chromo *population, int inicio, int fin)
{
    int i, pos = inicio;

    for (i = inicio + 1; i < fin; i++)
    {

        if (population[i].fitness < population[pos].fitness)
        {
            pos = i;
        }
    }

    return pos;
}



void selectChampionship(Chromo *parents, Chromo *population, int N, int p)
{

    int j, i, c = 0;

    // shuffle(population, p, N); // check

    for (i = 0; i < p; i = i + 2)
    {
        if (population[i].fitness <= population[i + 1].fitness)
        {
            // parents[c] = population[i];
            for (j = 0; j < N; j++)
            {
                parents[c].config[j] = population[i].config[j];
            }
        }
        else
        {
            // parents[c] = population[i + 1];
            for (j = 0; j < N; j++)
            {
                parents[c].config[j] = population[i + 1].config[j];
            }
        }
        c++;
    }
}
