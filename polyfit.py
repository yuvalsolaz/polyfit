# !pip install descartes

from __future__ import (absolute_import, division, print_function)

import sys
import numpy as np
from shapely.wkt import loads
from shapely import affinity
from geopandas import GeoSeries, GeoDataFrame

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))

from collections import namedtuple
Params = namedtuple('Params',['x','y','alpha'])

max_iteration = 1000
learning_rate = .1
loss_threshold_count = 20


def draw(shape,color='black'):
    GeoSeries(shape).plot(ax=ax,color=color)
    plt.pause(0.05)

def draw_loss(poly,rect):
    GeoSeries(rect).plot(ax=ax,color='yellow')
    x = xor(poly=poly, rect=rect)
    GeoSeries(x).plot(ax=ax,color='black')
    plt.pause(0.05)


def xor(poly, rect='y'):
    intersection = poly.intersection(rect)
    nonoverlap = rect.difference(intersection)
    return nonoverlap

def loss(poly, rect):
    return np.float(xor(poly,rect).area)

# calculate gradient numericly:
def gradient(poly,rect):
    eps = Params(x=.1, y=.1, alpha=0.1)
    l = loss(poly,rect)
    return Params(x = l - loss(poly, affinity.translate(rect,eps.x, 0)),
                  y = l- loss(poly, affinity.translate(rect,0, eps.y)),
                  alpha = l- loss(poly, affinity.rotate(rect,eps.alpha, origin='centroid')))


def sgd(poly,cannonical_rect):


    # initialize minimum loss with current loss
    min_loss = sys.float_info.max

    # initialize params :
    # x,y = aligned with polygon centroid
    # alpha = zero
    rect = cannonical_rect
    curr_params = Params(x = poly.centroid.x - rect.centroid.x ,
                         y = poly.centroid.y - rect.centroid.y ,
                         alpha = 0.0)


    # sgd loop
    for i in range (max_iteration) :

        # calculates current loss
        curr_loss = loss(poly, rect)

        # check exit condition TODO loss_threshold_count:
        #if curr_loss - min_loss > .1:
        #    break

        # update minimum loss
        min_loss = np.minimum(curr_loss,min_loss)

        # calculate gradient :
        g = gradient(poly,rect)

        # draw current position & current loss:
        draw_loss(poly,rect)
        print (f'iteration {i} curr_loss: {curr_loss} minimum loss {min_loss} grad: {g}')

        # update parameters x , y , alpha
        curr_params = Params(x = curr_params.x - learning_rate * g.x,
                             y = curr_params.y - learning_rate * g.y,
                             alpha = curr_params.alpha - learning_rate * g.alpha)

        # translate rect:
        rect = affinity.translate(cannonical_rect, curr_params.x, curr_params.y)
        rect = affinity.rotate(rect, curr_params.alpha)




if __name__ == '__main__':

    poly = loads('POLYGON((-150 100, -50 150, 0 100, 110 160, 150 0, 50 -50, -100 -90, -150 100))')
    rect = loads('POLYGON((-150 -50, -150 50, 150 50, 150 -50, -150 -50))')

    draw(poly,color='blue')
    draw(rect,color='green')

    sgd(poly,rect)
