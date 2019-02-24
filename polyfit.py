# !pip install descartes

from __future__ import (absolute_import, division, print_function)

from shapely.wkt import loads
from shapely import affinity
from geopandas import GeoSeries, GeoDataFrame

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))


def draw(shape,color='black'):
    GeoSeries(shape).plot(ax=ax,color=color)
    plt.pause(0.05)

def xor(poly, rect='y'):
    intersection = poly.intersection(rect)
    nonoverlap = rect.difference(intersection)
    return nonoverlap

def loss(poly, rect):
    return xor(poly,rect).area

if __name__ == '__main__':
    poly = loads('POLYGON((-150 100, -50 150, 0 100, 110 160, 150 0, 50 -50, -100 -90, -150 100))')
    rect = loads('POLYGON((-100 -50, -100 50, 100 50, 100 -50, -100 -50))')
    draw(poly,color='blue')
    draw(rect,color='green')
    x = xor(poly=poly, rect=rect)
    draw(x)
    plt.pause(5)

    for i in range(90):
        rls = affinity.rotate(rect, i, origin='centroid')
        draw(shape=rls,color='yellow')
        x = xor(poly=poly, rect=rls)
        draw(x)
        print(loss(poly,rls))
