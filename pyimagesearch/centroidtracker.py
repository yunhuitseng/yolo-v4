# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0  # next unique object ID
        self.objects = OrderedDict()  # object ID -> (x, y)
        self.rect = OrderedDict()  # object ID -> (x, y, w, h)
        # object ID -> number of consecutive frames it has been marked as "disappeared"
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared  # 允許的最大連續幀數 -> 超過就註銷

    def register(self, centroid, boundary):
        self.objects[self.nextObjectID] = centroid
        self.rect[self.nextObjectID] = boundary
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.rect[objectID]

    def sorting(self):
        for objectID in self.objects.keys():
            min = objectID
            l = [self.objects.keys()]
            for j in l[objectID+1:]:
                if self.objects[j][0] < self.objects[min][0]:
                    min = j
            self.objects[objectID], self.objects[min] = self.objects[min], self.objects[objectID]
            self.rect[objectID], self.rect[min] = self.rect[min], self.rect[objectID]
            self.disappeared[objectID], self.disappeared[min] = self.disappeared[min], self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:  # if the list of input bounding box rectangles is empty
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1  # 紀錄大家消失了一幀時間(醜一)
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.rect

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputSize = np.zeros((len(rects), 4), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            w = int(endX - startX)
            h = int(endY - startY)
            inputCentroids[i] = (cX, cY)
            inputSize[i] = (startX, startY, w, h)

        # if we are currently not tracking any objects take the input centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputSize[i])

        # otherwise, are are currently tracking objects so we need to try to match the input centroids to existing object centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids) # 計算兩個點之間的距離
            
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows] 

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.rect[objectID] = inputSize[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared or self.objects[objectID][0] < 0:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    # 這裡開始魔改
                    if inputCentroids[col][0] > 100:
                        self.register(inputCentroids[col], inputSize[col])
        
        self.sorting()
        
        return self.objects, self.rect
