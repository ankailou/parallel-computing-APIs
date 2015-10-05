#include <iostream>
#include <map>
#include <cstdlib>
#include "lab1.h"
using namespace std;

/*****************************************************************************
 ******************* helper functions for populating boxes *******************
 *****************************************************************************/

void populateBoxes() {
    Box box;
    int id, x, y, height, width, numNeighbor, tmp, dsv;
    cin >> numBoxes >> numRows >> numCols;
    cin >> id;
    while ( id != TERMINATE ) {
        cin >> x >> y >> height >> width;
        setParameters(&box, x, y, height, width);
        for (int i = 0; i < NUM_SIDES; i++) {
            cin >> numNeighbor;
            int *neighbors;
            neighbors = (int *)malloc(sizeof(int)*numNeighbor);
            for (int j = 0; j < numNeighbor; j++) {
                cin >> tmp;
                neighbors[j] = tmp;
            }
            setNeighbors(&box, i, numNeighbor, &neighbors);
        }
        cin >> dsv;
        box.dsv = (float)dsv;
        Boxes[id] = box;
        cin >> id;
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
