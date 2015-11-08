#include "lab2.h"
#include <iostream>
#include <stdio.h>
#include "omp.h"
using namespace std;

/*****************************************************************************
 *********** helper functions for computing the convergence loops ************
 *****************************************************************************/

/**
 * function: convergenceLoop
 ***************************
 * loop 1: compute new dsv score for each box; loop 2: ommunicate dsv scores;
 *
 * updates: for each @box in @Boxes: @box.dsv = computeNewDSV(@box);
 */
void convergenceLoop(int numThreads) {
    omp_set_num_threads(numThreads);
    cout << "Expected number of threads: " << omp_get_max_threads() << endl;
    #pragma omp parallel
    {
    #pragma omp for
    for (int i = 0; i < numBoxes; i++) {
        computeNewDSV(&Boxes[i]);
        if (i == 0)
            printf("Actual number of threads: %d\n", omp_get_num_threads());
    }
    }
    for (int j = 0; j < numBoxes; j++)
        Boxes[j].dsv = Boxes[j].dsvNew;
}

/*****************************************************************************
 *************** helper function for computing new dsv values ****************
 *****************************************************************************/

/**
 * function: computeNewDSV
 *************************
 * for each side, average adjacent temperature = sum(intersection * dsv) / perimeter;
 *
 * param box: struct holding box information; parameters to be updated;
 * updates: @box.newDSV = @box.dsv - [AFFECT_RATE * (@box.dsv - avg_adjacent_temp)];
 */
void computeNewDSV(Box *box) {
    int perimeter = 0;
    float avgAdjacent = 0.0, offset = 0.0;
    int numNeighbors[NUM_SIDES] = {(*box).nTop, (*box).nLeft, (*box).nBottom, (*box).nRight};
    for (int i = 0; i < NUM_SIDES; i++) {
        if (numNeighbors[i] > 0)
            perimeter += (i % 2 == 0) ? (*box).w : (*box).h;
        avgAdjacent += computeSide(&(*box), i, numNeighbors[i]);
    }
    if (perimeter > 0)
        offset = AFFECT_RATE * ((*box).dsv - (avgAdjacent / (float)perimeter));
    (*box).dsvNew = (*box).dsv - offset;
}

/**
 * function: computeSide
 ***********************
 * compute average adjacent temperature for one side of @box;
 *
 * param box: struct holding box information;
 * param side: int representing side := { 0 : top, 1 : left, 2 : bottom, 3 : right };
 * param numNeighbors: int representing number of neighboring boxes to iterate;
 * returns: sum_{nbox in @Boxes}( intersect(nBox,@box) * nBox.dsv );
 */
float computeSide(Box *box, int side, int numNeighbors) {
    int is;
    float temp = 0.0;
    int *sides[4] = {(*box).topNeighbor, (*box).leftNeighbor, (*box).bottomNeighbor, (*box).rightNeighbor};
    for (int i = 0; i < numNeighbors; i++) {
        Box neighbor = Boxes[sides[side][i]];
        if (side % 2 == 0)
            is = intersect((*box).x, (*box).w, neighbor.x, neighbor.w);
        else
            is = intersect((*box).y, (*box).h, neighbor.y, neighbor.h);
        temp += neighbor.dsv * is;
    }
    return temp;
}

/**
 * function: intersect
 *********************
 * compute the length of the intersection between adjacent boxes;
 *
 * param s1: start index of side of box 1;
 * param l1: length of the side of box 1;
 * param s2: start index of side of box 2;
 * param l2: length of the side of box 2;
 * returns: l1 - [ segments of s2 + l2 not intersecting with box 1 ];
 */
int intersect(int s1, int l1, int s2, int l2) {
    int intersect = l1;
    if ( s2 > s1 )
        intersect -= (s2 - s1);
    if ((s2 + l2) < (s1 + l1))
        intersect -= ((s1 + l1) - (s2 + l2));
    return intersect;
}
