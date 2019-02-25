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
learning_rate = .01
loss_threshold_count = 20

'''
def draw(shape,color='black'):
    GeoSeries(shape).plot(ax=ax,color=    poly = loads('POLYGON((-150 100, -50 150, 0 100, 110 160, 150 0, 50 -50, -100 -90, -150 100))')
color)
    plt.pause(0.05)
'''

def draw_loss(poly,rect):
    plt.cla()
    GeoSeries(poly).plot(ax=ax,color='green')
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
    eps = Params(x=1, y=1, alpha=.1)
    l = loss(poly,rect)
    return Params(x = (loss(poly, affinity.translate(rect,eps.x, 0)) - l) / eps.x,
                  y = (loss(poly, affinity.translate(rect,0, eps.y)) - l) / eps.y,
                  alpha = (loss(poly, affinity.rotate(rect,eps.alpha, origin='centroid')) - l) / eps.alpha )


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

    min_loss_iteration = 0

    # sgd loop
    for i in range (max_iteration) :

        # calculates current loss
        curr_loss = loss(poly, rect)

        # update minimum point iteration count:
        min_loss_iteration = min_loss_iteration + 1 if np.abs(curr_loss - min_loss) < .01 else 0

        # check exit condition
        if curr_loss == 0 or min_loss_iteration > loss_threshold_count :
            input(f'local minimum found. \n current loss : {curr_loss} \n ok ?')
            break

        # update minimum loss
        min_loss = np.minimum(curr_loss,min_loss)

        # calculate gradient :
        g = gradient(poly,rect)

        # draw current position & current loss:
        draw_loss(poly,rect)
        print (f'iteration:{i} current loss:{curr_loss} min loss:{min_loss} min loss iteration:{min_loss_iteration} grad:{g}')

        # update parameters x , y , alpha
        curr_params = Params(x = curr_params.x - learning_rate * g.x,
                             y = curr_params.y - learning_rate * g.y,
                             alpha = curr_params.alpha - learning_rate * g.alpha)

        # translate rect:
        rect = affinity.translate(cannonical_rect, curr_params.x, curr_params.y)
        rect = affinity.rotate(rect, curr_params.alpha)


if __name__ == '__main__':

    poly = loads('POLYGON((-150 80, -50 120, 0 100, 110 110, 150 0, 50 -50, -100 -90, -150 80))')

    if len(sys.argv) < 3:
        print(f'usage: python {sys.argv[0]} <width> <height>')
        exit(1)
    w = int(sys.argv[1]) / 2
    h = int(sys.argv[2]) / 2

    rect = loads(f'POLYGON(({-w} {-h}, {-w} {h}, {w} {h}, {w} {-h}, {-w} {-h}))')

    sgd(poly,rect)
