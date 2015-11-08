#include <iostream>
#include <cstdlib>
#include <sstream>
#include <sys/time.h>
#include "lab2.h"
using namespace std;

/*****************************************************************************
 ***************************** global veriables ******************************
 *****************************************************************************/

int numBoxes, numRows, numCols, loops = 0;
float runtime = 0.0;
map<int,Box> Boxes;

/*****************************************************************************
 ******************************* main function *******************************
 *****************************************************************************/

/**
 * function: main
 ****************
 * populate boxes; loop until convergence condition; otherwise compute dsv
 *
 * param argv[1]: name of file to read box parameters
 */
int main(int argc, char* argv[]) {
    int numThreads;
    struct timeval t1, t2;
    populateBoxes(argv[1]);
    istringstream ss(argv[2]);
    ss >> numThreads;
    while ( !convergenceCondition() ) {
        loops++;
        gettimeofday(&t1, NULL);
        convergenceLoop(numThreads);
        gettimeofday(&t2, NULL);
        float diff = ((unsigned long long)t2.tv_usec - (unsigned long long)t1.tv_usec) / 1000000.0;
        //cout << "Loop " << loops << ": " << diff << " seconds!" << endl;
        runtime += diff;
    }
    cout << "Convergence Loops Complete. Freeing Memory..." << endl;
    freeBoxes();
    cout << "Number of Convergence Loops: " << loops << endl;
    cout << "Total Program Running Time:  " << runtime << " seconds!" << endl;
}

/*****************************************************************************
 *************** helper function to determine when convergent ****************
 *****************************************************************************/

/**
 * function: convergenceCondition
 ********************************
 * compute difference between max and min dsv value in @Boxes
 *
 * returns: true/false value based on convergence condition
 */
int convergenceCondition() {
    float max = Boxes[0].dsv;
    float min = Boxes[0].dsv;
    for (int i = 1; i < numBoxes; i++) {
        if ( Boxes[i].dsv > max ) { max = Boxes[i].dsv; }
        if ( Boxes[i].dsv < min ) { min = Boxes[i].dsv; }
    }
    int con = ((max - min) <= (CONVERGENCE_CONST * max)) ? 1 : 0;
    if (con) cout << "Final Max: " << max << "; Final Min: " << min << endl;
    return con;
}

/*****************************************************************************
 *********************** helper function to free memory **********************
 *****************************************************************************/

/**
 * function: freeBoxes
 *********************
 * free all malloc'd memory
 */
void freeBoxes() {
    for (int i = 0; i < numBoxes; i++) {
        free(Boxes[i].topNeighbor);
        free(Boxes[i].bottomNeighbor);
        free(Boxes[i].leftNeighbor);
        free(Boxes[i].rightNeighbor);
    }
}
