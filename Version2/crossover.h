void inicializaCruzaHijos(int *c1, int *c2, int N)
{
    for (int m = 0; m < N; m++)
    {
        c1[m] = -1;
        c2[m] = -1;
    }
}

void cruzaInicializaPadres(int *p1, int *p2, int *ind1, int *ind2, int N)
{
    for (int m = 0; m < N; m++)
    {
        p1[m] = ind1[m];
        p2[m] = ind2[m];
    }
}

void cruzaCopiaMedio(int *c1, int *c2, int *p1, int *p2, int inicio, int fin)
{
    for (int a = inicio; a < fin; a++)
    {
        c1[a] = p2[a];
        c2[a] = p1[a];
    }
}

void cruzaCopiaExtremo(int *c1, int *c2, int *p1, int *p2, int inicio, int fin, int N, int inimedio, int finmedio)
{
    int flag1;
    int flag2;
    int pos1, pos2;
    for (int a = inicio; a < fin; a++)
    {
        flag1 = 0;
        flag2 = 0;
        pos1 = p1[a];
        pos2 = p2[a];
        for (int b = inimedio; b < finmedio; b++)
        {
            if (pos1 == c1[b])
            {
                flag1 = 1;
            }
            if (pos2 == c2[b])
            {
                flag2 = 1;
            }
        }
        if (!flag1)
        {
            c1[a] = pos1;
        }
        if (!flag2)
        {
            c2[a] = pos2;
        }
    }
}

// Partially Mapped Crossover
void Crossover(Chromo *parents, Chromo *population, int N, int inicio,int fin)
{


    // hijos
    int *c1 = (int *)malloc(sizeof(int) * N);
    int *c2 = (int *)malloc(sizeof(int) * N);
    // padres
    int *p1 = (int *)malloc(sizeof(int) * N);
    int *p2 = (int *)malloc(sizeof(int) * N);
    int flag1;

    int k = N / 3;
    int posnp = fin;

    for (int n = inicio; (n + 1) < fin; n = n + 2)
    {

        flag1 = 0;


        inicializaCruzaHijos(c1, c2, N);

        // inicializo los padres
        cruzaInicializaPadres(p1, p2, parents[n].config, parents[n + 1].config, N);

        cruzaCopiaMedio(c1, c2, p1, p2, k, (N - k));

        cruzaCopiaExtremo(c1, c2, p1, p2, 0, k, N, k, (N - k));

        cruzaCopiaExtremo(c1, c2, p1, p2, (N - k), N, N, k, (N - k));

        int count, co;
        for (int a = 0; a < N; a++)
        {
            flag1 = 0;
            count = 0;
            co = 0;
            while ((!flag1) && count < N)
            {
                if (a == c1[count])
                {
                    flag1 = 1;
                }
                count++;
            }

            if (!flag1)
            {
                while ((c1[co] != -1) && (co < N))
                {
                    co++;
                }
                c1[co] = a;
            }
            flag1 = 0;
            count = 0;
            co = 0;
            while ((!flag1) && count < N)
            {
                if (a == c2[count])
                {
                    flag1 = 1;
                }
                count++;
            }

            if (!flag1)
            {
                while ((c2[co] != -1) && (co < N))
                {
                    co++;
                }
                c2[co] = a;
            }
        }

        for (int i = 0; i < N; i++)
        {
            population[posnp].config[i] = c1[i];
            population[posnp + 1].config[i] = c2[i];
        }

        posnp = posnp + 2;
    }
}
