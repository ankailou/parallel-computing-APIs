#include "lab1.h"
using namespace std;

/*****************************************************************************
 *********** helper functions for computing the convergence loops ************
 *****************************************************************************/

void convergenceLoop() {
    for (int i = 0; i < numBoxes; i++) {
        computeNewDSV(&Boxes[i]);
    }
    for (int j = 0; j < numBoxes; j++) Boxes[j].dsv = Boxes[j].dsvNew;
}

/*****************************************************************************
 *************** helper function for computing new dsv values ****************
 *****************************************************************************/

void computeNewDSV(Box *box) {
    int perimeter = 0;
    float avgAdjacent = 0.0, offset = 0.0;
    int numNeighbors[NUM_SIDES] = {(*box).nTop, (*box).nLeft,
                                   (*box).nBottom, (*box).nRight};
    for (int i = 0; i < NUM_SIDES; i++) {
        if (numNeighbors[i] > 0) {
            perimeter += (i % 2 == 0) ? (*box).w : (*box).h;
        }
        avgAdjacent += computeSide(&(*box), i, numNeighbors[i]);
    }
    if (perimeter > 0)
        offset = 0.1 * ((*box).dsv - (avgAdjacent / (float)perimeter));
    (*box).dsvNew = (*box).dsv - offset;
}

float computeSide(Box *box, int side, int numNeighbors) {
    int is;
    float temp = 0.0;
    int *sides[4] = {(*box).topNeighbor, (*box).leftNeighbor,
                     (*box).bottomNeighbor, (*box).rightNeighbor};
    for (int i = 0; i < numNeighbors; i++) {
        Box neighbor = Boxes[sides[side][i]];
        float dsv = neighbor.dsv;
        if ( side % 2 == 0)
            is = intersect((*box).x, (*box).w, neighbor.x, neighbor.w);
        else
            is = intersect((*box).y, (*box).h, neighbor.y, neighbor.h);
        temp += dsv * is;
    } return temp;
}

int intersect(int s1, int l1, int s2, int l2) {
    int intersect = l1;
    if ( s2 > s1 ) { intersect -= (s2 - s1); }
    if ( (s2 + l2) < (s1 + l1) ) { intersect -= ((s1 + l1) - (s2 + l2)); }
    return intersect;
}
