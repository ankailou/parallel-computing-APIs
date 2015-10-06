#include <iostream>
#include <sys/time.h>
#include "lab1.h"
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

int main() {
    struct timeval t1, t2;
    populateBoxes();
    for (;;) {
        if ( convergenceCondition() ) {
            break;
        } else {
            loops++;
            gettimeofday(&t1, NULL);
            convergenceLoop();
            gettimeofday(&t2, NULL);
            float diff = ((float)t2.tv_usec - (float)t1.tv_usec) / (float)1000000;
            cout << "Loop " << loops << ": " << diff << " seconds!" << endl;
            runtime += diff;
        }
    }
    cout << "Number of Convergence Loops: " << loops << endl;
    cout << "Total Program Running Time:  " << runtime << " seconds!" << endl;
}

/*****************************************************************************
 *************** helper function to determine when convergent ****************
 *****************************************************************************/

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
