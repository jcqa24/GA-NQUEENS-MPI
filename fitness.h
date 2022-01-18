void calFit(Chromo *population, int N, int inicio, int fin)
{
    int errores;
    int k, i, j;
    for (k = inicio; k < fin; k++)
    {
        errores = 0;
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (i != j)
                {
                    if ((population[k].config[i] - i) == (population[k].config[j] - j))
                    {
                        errores++;
                    }
                    if ((population[k].config[i] + i) == (population[k].config[j] + j))
                    {
                        errores++;
                    }
                }
            }
        }

        population[k].fitness = errores;
    }
}


void copyBest(Chromo *Best, Chromo local, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        Best->config[i] = local.config[i];
    }
    Best->fitness = local.fitness;
}
