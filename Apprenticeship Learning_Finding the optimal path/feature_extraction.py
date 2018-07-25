# -*- coding: utf-8 -*-
import pickle
import math
data = pickle.load(open('training_data.p', 'rb'))

def feature_set_1(point,grid):
    f1=[]
    f2=[]
    f3=[]
    f4=[]
    x, y = point[0], point[1]
    grids = []

    for i in range(0, len(data)): #getting all grid in grids list
        grids.append(data[i][0])

    def down(grid): #feature 3
        for count in range(0, 7):
            if 1 <= x - count <= 7:
                if (grids[grid])[7 - x + count][y] != 2:
                    f3.append(1)

        return f3
    down(grid)

    def up(grid): #feature 1
        for count in range(0, 7):
            if 1 <= x + count <= 7:
                if (grids[grid])[7 - x - count][y] != 2:
                    f1.append(1)
        return f1
    up(grid)

    def right(grid): #feature 2
        for count in range(0, 7):
            if 1<= y + count <= 7:
                if (grids[grid])[7-x][y + count] != 2:
                    f2.append(1)
        return f2
    right(grid)

    def left(grid): #feature 4
        for count in range(0, 7):
            if 1 <= y - count <= 7:
                if (grids[grid])[7-x][y - count] != 2:
                    f4.append(1)

        return f4
    left(grid)
    feature_set1= [sum(f1),sum(f2),sum(f3),sum(f4)]
    return feature_set1
def feature_set_2(point, grid):
    """Computes the distance from the agent to the light blue squares
       one unit away from it
       Parameters
       ----------
       point: (x,y) the position of the agent in the grid
       grid: 2d array representing a grid
       Returns
       -------
       [f5,f6,f7,f8]: a list of features in the given format
       """
    f5 = []
    f6 = []
    f7 = []
    f8 = []

    x,y = point[0], point [1]
    grids = []
    for i in range(0, len(data)):
        grids.append(data[i][0])

    def f5_(grid): #feature 5
        if (grids[grid])[(7 - x) - 1][y] == 1:
            f5.append(1)
        if (grids[grid])[(7- x) - 1][y + 1] == 1:
            f5.append(1)
        if (grids[grid])[(7- x) - 1][y - 1] == 1:
            f5.append(1)
        return f5
    f5_(grid)
    def f6_(grid): #feature 6
        if (grids[grid])[7-x][y + 1] == 1:
            f6.append(1)
        if (grids[grid])[7-x - 1][y +1] == 1:
            f6.append(1)
        if (grids[grid])[7-x + 1][y+1] == 1:
            f6.append(1)
        return f6
    f6_(grid)

    def f7_(grid): #feature 7
        if (grids[grid])[7- x + 1][y] == 1:
            f7.append(1)
        if (grids[grid])[7- x + 1][y + 1] == 1:
            f7.append(1)
        if (grids[grid])[7- x + 1][y - 1] == 1:
            f7.append(1)
        return f7
    f7_(grid)

    def f8_(grid): #feature 8
        if (grids[grid])[7- x][y - 1] == 1:
            f8.append(1)
        if (grids[grid])[7- x + 1][y - 1] == 1:
            f8.append(1)
        if (grids[grid])[7- x - 1][y - 1] == 1:
            f8.append(1)
        return f8
    f8_(grid)

    feature_set2 = [sum(f5), sum(f6), sum(f7), sum(f8)]
    return feature_set2
def feature_set_3(point, grid):
    feature_set3=[]
    x,y = point[0],point[1]
    distance = (x - 1) ** 2 + (y - 6) ** 2 #feature 9
    feature_set3.append(math.sqrt(distance))
    return feature_set3
def getY():
    Y = []
    kv = []
    for i in range(0, len(data)):
        for j in data[i][1].iteritems():
            kv.append(j)
    chunks = [kv[x:x + 64] for x in xrange(0, len(kv), 64)]
    for i in range(len(chunks)):
        chunks[i].sort()
    dc = [rows for sublist in chunks for rows in sublist]
    for i in range(len(dc)):
        for x in range(1, 7):
            for y in range(1, 7):
                if dc[i][0][0] == x:
                    if dc[i][0][1] == y:
                        Y.append(dc[i][1])
    return Y