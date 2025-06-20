/************************************************************************
*
*  lap.js -- ported to javascript from

   lap.cpp
   version 1.0 - 4 September 1996
   author: Roy Jonker @ MagicLogic Optimization Inc.
   e-mail: roy_jonker@magiclogic.com

   Code for Linear Assignment Problem, according to

   "A Shortest Augmenting Path Algorithm for Dense and Sparse Linear
    Assignment Problems," Computing 38, 325-340, 1987

   by

   R. Jonker and A. Volgenant, University of Amsterdam.

*
   PORTED TO JAVASCRIPT 2017-01-02 by Philippe Riviere(fil@rezo.net)
   CHANGED 2016-05-13 by Yang Yong(yangyongeducation@163.com) in column reduction part according to
   matlab version of LAPJV algorithm(Copyright (c) 2010, Yi Cao All rights reserved)--
   https://www.mathworks.com/matlabcentral/fileexchange/26836-lapjv-jonker-volgenant-algorithm-for-linear-assignment-problem-v3-0:
*
*************************************************************************/

/* This function is the jv shortest augmenting path algorithm to solve the assignment problem */
function lap(dim, cost) {
  // input:
  // dim        - problem size
  // cost       - cost callback (or matrix)

  // output:
  // rowsol     - column assigned to row in solution
  // colsol     - row assigned to column in solution
  // u          - dual variables, row reduction numbers
  // v          - dual variables, column reduction numbers

  // convert the cost matrix (old API) to a callback (new API)
  if (typeof cost === "object") {
    var cost_matrix = cost;
    cost = function(i, j) {
      return cost_matrix[i][j];
    };
  }

  var sum = 0;
  {
    let i1, j1;
    for (i1 = 0; i1 < dim; i1++) {
      for (j1 = 0; j1 < dim; j1++)
        sum += cost(i1, j1);
    }
  }
  const BIG = 10000 * (sum / dim);
  const epsilon = sum / dim / 10000;
  const rowsol = new Int32Array(dim),
    colsol = new Int32Array(dim),
    u = new Float64Array(dim),
    v = new Float64Array(dim);
  let unassignedfound;
  /* row */
  let i, imin, numfree = 0, prvnumfree, f, i0, k, freerow; // *pred, *free
  /* col */
  let j, j1, j2, endofpath, last, low, up; // *collist, *matches
  /* cost */
  let min, h, umin, usubmin, v2; // *d

  const free = new Int32Array(dim); // list of unassigned rows.
  const collist = new Int32Array(dim); // list of columns to be scanned in various ways.
  const matches = new Int32Array(dim); // counts how many times a row could be assigned.
  const d = new Float64Array(dim); // 'cost-distance' in augmenting path calculation.
  const pred = new Int32Array(dim); // row-predecessor of column in augmenting/alternating path.

  // init how many times a row will be assigned in the column reduction.
  for (i = 0; i < dim; i++)
    matches[i] = 0;

  // COLUMN REDUCTION
  for (
    j = dim;
    j--; // reverse order gives better results.

  ) {
    // find minimum cost over rows.
    min = cost(0, j);
    imin = 0;
    for (i = 1; i < dim; i++)
      if (cost(i, j) < min) {
        min = cost(i, j);
        imin = i;
      }
    v[j] = min;
    if (++matches[imin] == 1) {
      // init assignment if minimum row assigned for first time.
      rowsol[imin] = j;
      colsol[j] = imin;
    } else if (v[j] < v[rowsol[imin]]) {
      j1 = rowsol[imin];
      rowsol[imin] = j;
      colsol[j] = imin;
      colsol[j1] = -1;
    } else colsol[j] = -1; // row already assigned, column not assigned.
  }

  // REDUCTION TRANSFER
  for (i = 0; i < dim; i++) {
    if (
      matches[i] == 0 // fill list of unassigned 'free' rows.
    )
      free[numfree++] = i;
    else if (matches[i] == 1) {
      // transfer reduction from rows that are assigned once.
      j1 = rowsol[i];
      min = BIG;
      for (j = 0; j < dim; j++)
        if (j != j1)
          if (cost(i, j) - v[j] < min + epsilon) min = cost(i, j) - v[j];
      v[j1] = v[j1] - min;
    }
  }

  // AUGMENTING ROW REDUCTION
  let loopcnt = 0; // do-loop to be done twice.
  do {
    loopcnt++;

    // scan all free rows.
    // in some cases, a free row may be replaced with another one to be scanned next.
    k = 0;
    prvnumfree = numfree;
    numfree = 0; // start list of rows still free after augmenting row reduction.
    while (k < prvnumfree) {
      i = free[k];
      k++;

      // find minimum and second minimum reduced cost over columns.
      umin = cost(i, 0) - v[0];
      j1 = 0;
      usubmin = BIG;
      for (j = 1; j < dim; j++) {
        h = cost(i, j) - v[j];
        if (h < usubmin)
          if (h >= umin) {
            usubmin = h;
            j2 = j;
          } else {
            usubmin = umin;
            umin = h;
            j2 = j1;
            j1 = j;
          }
      }

      i0 = colsol[j1];
      if (umin < usubmin + epsilon)
        //         change the reduction of the minimum column to increase the minimum
        //         reduced cost in the row to the subminimum.
        v[j1] = v[j1] - (usubmin + epsilon - umin);
      else if (i0 > -1) {
        // minimum and subminimum equal.
        // minimum column j1 is assigned.
        // swap columns j1 and j2, as j2 may be unassigned.
        j1 = j2;
        i0 = colsol[j2];
      }

      // (re-)assign i to j1, possibly de-assigning an i0.
      rowsol[i] = j1;
      colsol[j1] = i;

      if (i0 > -1)
        if (umin < usubmin)
          // minimum column j1 assigned earlier.
          // put in current k, and go back to that k.
          // continue augmenting path i - j1 with i0.
          free[--k] = i0;
        else
          // no further augmenting reduction possible.
          // store i0 in list of free rows for next phase.
          free[numfree++] = i0;
    }
  } while (loopcnt < 2); // repeat once.

  // AUGMENT SOLUTION for each free row.
  for (f = 0; f < numfree; f++) {
    freerow = free[f]; // start row of augmenting path.

    // Dijkstra shortest path algorithm.
    // runs until unassigned column added to shortest path tree.
    for (j = dim; j--; ) {
      d[j] = cost(freerow, j) - v[j];
      pred[j] = freerow;
      collist[j] = j; // init column list.
    }

    low = 0; // columns in 0..low-1 are ready, now none.
    up = 0; // columns in low..up-1 are to be scanned for current minimum, now none.
    // columns in up..dim-1 are to be considered later to find new minimum,
    // at this stage the list simply contains all columns
    unassignedfound = false;
    do {
      if (up == low) {
        // no more columns to be scanned for current minimum.
        last = low - 1;

        // scan columns for up..dim-1 to find all indices for which new minimum occurs.
        // store these indices between low..up-1 (increasing up).
        min = d[collist[up++]];
        for (k = up; k < dim; k++) {
          j = collist[k];
          h = d[j];
          if (h <= min) {
            if (h < min) {
              // new minimum.
              up = low; // restart list at index low.
              min = h;
            }
            // new index with same minimum, put on undex up, and extend list.
            collist[k] = collist[up];
            collist[up++] = j;
          }
        }
        // check if any of the minimum columns happens to be unassigned.
        // if so, we have an augmenting path right away.
        for (k = low; k < up; k++)
          if (colsol[collist[k]] < 0) {
            endofpath = collist[k];
            unassignedfound = true;
            break;
          }
      }

      if (!unassignedfound) {
        // update 'distances' between freerow and all unscanned columns, via next scanned column.
        j1 = collist[low];
        low++;
        i = colsol[j1];
        h = cost(i, j1) - v[j1] - min;

        for (k = up; k < dim; k++) {
          j = collist[k];
          v2 = cost(i, j) - v[j] - h;
          if (v2 < d[j]) {
            pred[j] = i;
            if (v2 == min)
              if (colsol[j] < 0) {
                // new column found at same minimum value
                // if unassigned, shortest augmenting path is complete.
                endofpath = j;
                unassignedfound = true;
                break;
              } else {
                // else add to list to be scanned right away.
                collist[k] = collist[up];
                collist[up++] = j;
              }
            d[j] = v2;
          }
        }
      }
    } while (!unassignedfound);

    // update column prices.
    for (k = last + 1; k--; ) {
      j1 = collist[k];
      v[j1] = v[j1] + d[j1] - min;
    }

    // reset row and column assignments along the alternating path.
    do {
      i = pred[endofpath];
      colsol[endofpath] = i;
      j1 = endofpath;
      endofpath = rowsol[i];
      rowsol[i] = j1;
    } while (i != freerow);
  }

  // calculate optimal cost.
  let lapcost = 0;
  for (i = dim; i--; ) {
    j = rowsol[i];
    u[i] = cost(i, j) - v[j];
    lapcost = lapcost + cost(i, j);
  }

  return {
    cost: lapcost,
    row: rowsol,
    col: colsol,
    u: u,
    v: v
  };
}