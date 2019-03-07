# !pip install descartes

from __future__ import (absolute_import, division, print_function)

import sys
from shapely.wkt import loads
from shapely import affinity

from collections import namedtuple
Params = namedtuple('Params',['x','y','alpha'])

'''
gradient decent hyper params :
'''
max_iteration = 1000
learning_rate_xy = .1
learning_rate_alpha = .5
momentom = .9
loss_threshold_count = 20

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


def sgd(poly,cannonical_rect, init_params=Params(x=0, y=0, alpha=0)):
    # initialize minimum loss with current loss
    min_loss = sys.float_info.max
    min_loss_index = 0

    # initialize params :
    # x,y = aligned with polygon centroid
    # alpha = zero
    rect = cannonical_rect
    curr_params = init_params
    v = Params(x=0, y=0, alpha=0)


    # sgd loop TODO : add momentom
    for i in range (max_iteration) :

        # calculates current loss
        curr_loss = loss(poly, rect)

        # update minimum loss iteration index:
        if (min_loss - curr_loss) > 1e-2 :
            min_loss_index = i
            min_loss = curr_loss

        # calculate gradient :
        g = gradient(poly,rect)

        # calculate valocity
        v = Params(x = momentom*v.x + (1-momentom) * g.x,
                        y = momentom * v.y + (1 - momentom) * g.y,
                        alpha = momentom * v.alpha + (1 - momentom) * g.alpha)

        # draw current position & current loss:
        draw(poly, rect)
        print(f'iteration:{i} current loss:{curr_loss} min loss:{min_loss} min loss index:{min_loss_index} v:{v}')

        # update parameters x , y , alpha
        curr_params = Params(x =curr_params.x - learning_rate_xy * v.x,
                             y =curr_params.y - learning_rate_xy * v.y,
                             alpha =curr_params.alpha - learning_rate_alpha * v.alpha)

        # translate rect:
        rect = affinity.translate(cannonical_rect, curr_params.x, curr_params.y)
        rect = affinity.rotate(rect, curr_params.alpha)

        # check stopping condition
        if curr_loss == 0 or i - min_loss_index > loss_threshold_count:
            break

    return min_loss, min_loss_index


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

    vertex = np.diff(poly.boundary.coords.xy)
    vertex_azimuth = np.arctan2(vertex[1],vertex[0]) * 180.0 / np.pi
    for alpha in vertex_azimuth:
        # init params :
        init_params = Params(x=poly.centroid.x - rect.centroid.x,
                             y=poly.centroid.y - rect.centroid.y,
                             alpha=alpha)

        # draw init position:
#        draw(poly, affinity.rotate(affinity.translate(rect, init_params.x, init_params.y),init_params.alpha))
#       input(f'init params:{init_params} \n ok ?')
#        continue

        res = sgd(poly,rect, init_params=init_params)

        if res[0] < 1e-3:
            break
    input(f'local minimum found. \n minimum loss:{res[0]} after {res[1]} iterations. \n initial alpha:{alpha} \n ok ?')


    #res = find_min(poly,rect)
    #input(f'minimize results:{res} \n ok ?')

