import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas


def show_box_plot():
    box_plot = dict()
    data_type = ['lw', 'ld', 'rw', 'rd']

    class_type = ['L', 'B', 'R']
    for x in ['L', 'B', 'R']:
        box_plot[x] = dict()
        for t in data_type:
            box_plot[x][t] = list()
    for line in file:
        data = line.split(',')
        for index, t in enumerate(data_type, start=1):
            box_plot[data[0]][t].append(int(data[index]))
    fig, axs = plt.subplots(3, sharex=True)
    for index, x in enumerate(class_type, start=0):
        data_to_plot = [box_plot[x]['lw'], box_plot[x]['ld'],box_plot[x]['rw'], box_plot[x]['rd']]
        axs[index].boxplot(data_to_plot)
        axs[index].set_title('class {}'.format(x))
        axs[index].set_xticklabels(['left weight', 'left distance', 'right weight', 'right distance'])
    plt.show()


def show_scatter_plot():
    scatter_plot = dict()
    for x in ['L', 'B', 'R']:
        scatter_plot[x] = list()
    for line in file:
        data = line.split(',')
        scatter_plot[data[0]].append((int(data[1]) * int(data[2]), int(data[3]) * int(data[4])))
        for x in ['L', 'B', 'R']:
            plt.scatter([lis[0] for lis in scatter_plot[x]], [lis[1] for lis in scatter_plot[x]],
                        label='class {}'.format(x))
    plt.ylabel('left weight * left distance')
    plt.xlabel('right weight * right distance')
    plt.title('scatter plot')
    plt.show()




#file = pd.read_csv('data.csv')
file = open(r"data.csv")

print(file)

#print("len file is:",len(file))



choice = input("enter b for box plot and s for scatter plot: ")
if choice == 'b':
    show_box_plot()
elif choice == 's':
    show_scatter_plot()

