
void mutation(Chromo *population, int prob, int N, int inicio, int fin)
{
    int aux, i, p1 = 0, p2 = 0;
    for (i = inicio; i < fin; i++)
    {
        srand(time(NULL));
        if (rand() % (101) <= prob)
        {
            do
            {
                p1 = rand() % (N + 1);
                p2 = rand() % (N + 1);
            } while (p1 == p2);
            aux = population[i].config[p1];
            population[i].config[p1] = population[i].config[p2];
            population[i].config[p2] = aux;
        }
    }
}
