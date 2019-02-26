# !pip install descartes

from __future__ import (absolute_import, division, print_function)

import sys
from shapely.wkt import loads
from shapely import affinity

from collections import namedtuple
Params = namedtuple('Params',['x','y','alpha'])

'''
    graphics display module   
'''
from geopandas import GeoSeries
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))

def draw(poly, rect):
    plt.cla()
    GeoSeries(poly).plot(ax=ax,color='green')
    GeoSeries(rect).plot(ax=ax,color='blue')
    x = diff(poly=poly, rect=rect)
    GeoSeries(x).plot(ax=ax,color='black')
    plt.pause(0.05)


def diff(poly, rect='y'):
    intersection = poly.intersection(rect)
    nonoverlap = rect.difference(intersection)
    return nonoverlap

def loss(poly, rect):
    return diff(poly, rect).area

# calculate gradient numericly:
def gradient(poly,rect):
    eps = Params(x=.1, y=.1, alpha=.1)
    l = loss(poly,rect)
    return Params(x = (loss(poly, affinity.translate(rect,eps.x, 0)) - l) / eps.x,
                  y = (loss(poly, affinity.translate(rect,0, eps.y)) - l) / eps.y,
                  alpha = (loss(poly, affinity.rotate(rect,eps.alpha, origin='centroid')) - l) / eps.alpha )

max_iteration = 1000
learning_rate = .1
loss_threshold_count = 40

def sgd(poly,cannonical_rect):
    # initialize minimum loss with current loss
    min_loss = sys.float_info.max
    min_loss_index = 0

    # initialize params :
    # x,y = aligned with polygon centroid
    # alpha = zero
    rect = cannonical_rect
    curr_params = Params(x = poly.centroid.x - rect.centroid.x ,
                         y = poly.centroid.y - rect.centroid.y ,
                         alpha = 0.0)

    # sgd loop TODO : add momentom
    for i in range (max_iteration) :

        # calculates current loss
        curr_loss = loss(poly, rect)

        # update minimum loss iteration index:
        if (min_loss - curr_loss) > 1e-5 :
            min_loss_index = i
            min_loss = curr_loss

        # check stopping condition
        if curr_loss == 0 or i - min_loss_index > loss_threshold_count :
            input(f'local minimum found. \n minimum loss:{min_loss} after {min_loss_index} iterations \n ok ?')
            break

        # calculate gradient :
        g = gradient(poly,rect)

        # draw current position & current loss:
        draw(poly, rect)
        print (f'iteration:{i} current loss:{curr_loss} min loss:{min_loss} min loss iteration:{min_loss_index} grad:{g}')

        # update parameters x , y , alpha
        curr_params = Params(x = curr_params.x - learning_rate * g.x,
                             y = curr_params.y - learning_rate * g.y,
                             alpha = curr_params.alpha - learning_rate * g.alpha)

        # translate rect:
        rect = affinity.translate(cannonical_rect, curr_params.x, curr_params.y)
        rect = affinity.rotate(rect, curr_params.alpha)

import numpy as np
from scipy.optimize import minimize

def find_min(poly,rect):

    def minimize_function(x):
        # draw current position & current loss:
        r = affinity.rotate(affinity.translate(rect, x[0], x[1]), x[2])
        l = loss(poly,r)
        print (f'x:{x[0]} {x[1]} {x[2]} loss:{l}')
        draw(poly,r)


        return loss(poly,affinity.rotate(affinity.translate(rect, x[0], x[1]), x[2]))

    x0 =  np.array([poly.centroid.x - rect.centroid.x ,poly.centroid.y - rect.centroid.y ,0.0])
    res = minimize(minimize_function, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
    print (f'minimized params: {res.x}')
    return res


if __name__ == '__main__':

    poly = loads('POLYGON((-150 140, -50 140,0 140 , 10 100, 110 110, 150 0, 50 -50, -100 -90, -150 140))')

    if len(sys.argv) < 3:
        print(f'usage: python {sys.argv[0]} <width> <height>')
        exit(1)
    w = int(sys.argv[1]) / 2
    h = int(sys.argv[2]) / 2

    rect = loads(f'POLYGON(({-w} {-h}, {-w} {h}, {w} {h}, {w} {-h}, {-w} {-h}))')

    # sgd(poly,rect)

    res = find_min(poly,rect)
    input(f'minimize results:{res} \n ok ?')

