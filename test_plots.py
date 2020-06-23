#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
Module responsible for provide custom plot functions for Promoter Prediction Project.

@ide: PyCharm
@author: Lauro Ângelo Gonçalves de Moraes
@contact: lauromoraes@ufop.edu.br
@created: 16/06/2020
"""

# import plotly.graph_objects as go
# fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
# fig.show()

def main():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    # import matplotlib.transforms as mtransforms

    x, y = np.random.random((2, 100)) * 2
    x = [0.8120561124, 0.7522180716, 0.8335708097, 0.8522079373, 0.7960447592]
    y = [0.8522079373, 0.8335708097, 0.8707802583, 0.907782657, 0.8335708097]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c='black')
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    plt.show()


if __name__ == "__main__":
    main()