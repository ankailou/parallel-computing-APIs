#include <map>
using namespace std;

/*****************************************************************************
 ******************************** constants **********************************
 *****************************************************************************/

#define NUM_SIDES 4
#define CONVERGENCE_CONST 0.1
#define AFFECT_RATE 0.1
#define TERMINATE -1

/*****************************************************************************
 ******************* class definitions & methods for Box *********************
 *****************************************************************************/

typedef struct Room {
    int x, y, h, w;
    int nTop, nLeft, nRight, nBottom;
    int *topNeighbor;
    int *leftNeighbor;
    int *rightNeighbor;
    int *bottomNeighbor;
    float dsv;
    float dsvNew;
} Box;

/*****************************************************************************
 ****************************** global variables *****************************
 *****************************************************************************/

extern int numBoxes, numRows, numCols, loops;
extern float runtime;
extern map<int,Box> Boxes;

/*****************************************************************************
 *************************** function prototypes *****************************
 *****************************************************************************/

int convergenceCondition();
void freeBoxes();

void populateBoxes(char*);
void setParameters(Box*, int, int, int, int);
void setNeighbors(Box*, int, int, int**);

void convergenceLoop(int);

void *computeNewDSV(void*);
float computeSide(Box*,int,int);
int intersect(int, int, int, int);
