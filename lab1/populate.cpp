#include <fstream>
#include <cstdlib>
#include "lab1.h"
using namespace std;

/*****************************************************************************
 ******************* helper functions for populating boxes *******************
 *****************************************************************************/

/**
 * function: populateBoxes
 *************************
 * read parameters from @file and populate map @Boxes with box structs;
 *
 * param file: pointer to file to read box parameters;
 * updates: @Boxes with struct box objects containing box parameters;
 */
void populateBoxes(char* file) {
    ifstream infile;
    int id, x, y, height, width, numNeighbor, tmp;
    float dsv;
    infile.open(file);
    infile >> numBoxes >> numRows >> numCols;
    infile >> id;
    while ( id != TERMINATE ) {
        Box box;
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

/**
 * function: setParameters
 *************************
 * set parameters of struct @box to other arguments;
 *
 * param box: struct holding box information; parameters to be updated;
 * param x: x-coordinate of the upper-leftmost corner of @box;
 * param y: y-coordinate of the upper-leftmost corner of @box;
 * param height: height of @box;
 * param width: width of @box;
 * updates: members variables of @box in @Boxes;
 */
void setParameters(Box *box, int x, int y, int height, int width) {
    (*box).x = x;
    (*box).y = y;
    (*box).h = height;
    (*box).w = width;
}

/**
 * function: setNeighbors
 ************************
 * set neighbor counter and neighbor id array of side @side in @box
 *
 * param box: struct holding box information; parameters to be updated;
 * param side: int representing side := { 0 : top, 1 : bottom, 2 : left, 3 : right}
 * param numNeighbors: int representing number of neighbor boxes on side @side
 * param neighbors: int array of ids to neighbor boxes on side @side
 * updates: member variables of @box in @Boxes according to @side
 */
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
