void InitConf(Chromo *population, int N, int inicio, int fin)
{

    int pos;
    int i, j, k;

    for (k = inicio; k < fin; k++)
    {
        population[k].config = (int *)malloc(sizeof(int) * N);
        for (j = 0; j < N; j++)
        {
            population[k].config[j] = -1;
        }
        i = 0;
        while (i < N)
        {
            pos = rand() % (N); // pos en el arreglo
            if ((population[k].config[pos] == -1) && (population[k].config[pos] != i))
            {
                population[k].config[pos] = i;
                i++;
            }
        }
    }
}

void reservaMemoria(Chromo *population, Chromo *parents, int p, int np, int N)
{
    int i;
    for (i = 0; i < p; i++)
    {
        population[i].config = (int *)malloc(sizeof(int) * N);
    }

    for (i = 0; i < np; i++)
    {
        parents[i].config = (int *)malloc(sizeof(int) * N);
    }
}
