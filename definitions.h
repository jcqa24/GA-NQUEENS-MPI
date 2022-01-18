struct Chromosome
{
    int fitness;
    int *config;
};
typedef struct Chromosome Chromo;

void ip(Chromo *, int , int );
void Insertion_sort(Chromo *, int );
void InitConf(Chromo *, int , int , int );
void shuffle(Chromo *, int , int );
int BuscaMin(Chromo *, int , int );
void inicializaCruzaHijos(int *, int *, int );
void cruzaInicializaPadres(int *, int *, int *, int *, int );
void cruzaCopiaMedio(int *, int *, int *, int *, int , int );
void cruzaCopiaExtremo(int *, int *, int *, int *, int , int , int , int , int );
void Crossover(Chromo *, Chromo *, int , int ,int );
void selectChampionship(Chromo *, Chromo *, int , int );
void imprimeTablero(int *,int );
void printConf(int *, int );
void calFit(Chromo *, int , int , int );
void mutation(Chromo *, int , int , int , int );
void copyBest(Chromo *, Chromo , int );
void reservaMemoria(Chromo *, Chromo *, int , int , int );
void confFinal(Chromo , int , clock_t,int);
int algoritmoGenetico(int , int , int , Chromo *, int , int , clock_t);
