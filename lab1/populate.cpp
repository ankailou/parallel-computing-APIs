#include <fstream>
#include <cstdlib>
#include "lab1.h"
using namespace std;

/*****************************************************************************
 ******************* helper functions for populating boxes *******************
 *****************************************************************************/

void populateBoxes(char* file) {
    ifstream infile;
    infile.open(file);
    Box box;
    int id, x, y, height, width, numNeighbor, tmp;
    float dsv;
    infile >> numBoxes >> numRows >> numCols;
    infile >> id;
    while ( id != TERMINATE ) {
        infile >> y >> x >> height >> width;
        setParameters(&box, x, y, height, width);
        for (int i = 0; i < NUM_SIDES; i++) {
            infile >> numNeighbor;
            int *neighbors;
            neighbors = (int *)malloc(sizeof(int)*numNeighbor);
            for (int j = 0; j < numNeighbor; j++) {
                infile >> tmp;
                neighbors[j] = tmp;
            }
            setNeighbors(&box, i, numNeighbor, &neighbors);
        }
        infile >> dsv;
        box.dsv = dsv;
        Boxes[id] = box;
        infile >> id;
    }
}

void setParameters(Box *box, int x, int y, int height, int width) {
    (*box).x = x;
    (*box).y = y;
    (*box).h = height;
    (*box).w = width;
}

void setNeighbors(Box *box, int side, int numNeighbor, int **neighbors) {
    switch (side) {
        case 0:
            (*box).nTop = numNeighbor;
            (*box).topNeighbor = (*neighbors);
            break;
        case 1:
            (*box).nBottom = numNeighbor;
            (*box).bottomNeighbor = (*neighbors);
            break;
        case 2:
            (*box).nLeft = numNeighbor;
            (*box).leftNeighbor = (*neighbors);
            break;
        case 3:
            (*box).nRight = numNeighbor;
            (*box).rightNeighbor = (*neighbors);
            break;
        default:
            break;
    }
}
