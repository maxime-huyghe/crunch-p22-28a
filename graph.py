import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

#notre tableau de tableaux de 3 valeurs avec mot,dov ou dod et increasing rate
a = ([['label',2,5],['énergie',2,7],['valeur',5,7],['coucou',9,3],['valeur',4,5],['coucou',-4,16],['coucou',-4,-5]])

x= ([])
y= ([])
xy=([])

#on les ajoute dans un tableau de x et de y
#on écrit les mots correspondant directement dans la boucle
for j in a:
    x.append(j[1])
    y.append(j[2])
    plt.text(j[1], j[2], j[0], fontsize=16)

#on fusionne les en un tableau xy
xy.append(x)
xy.append(y)

    
print('#############"')
print(x)
print(y)
print('#############"')
print(xy)

#librairie qui détermine les centres des clusters, il y en a 3 pour le noises, weak signals et strong signals
xy = np.dstack((x,y))
xy = xy[0]
model = KMeans(3).fit(xy)

#le nombre de couleurs en fonction du nombre de clusters
colors = [i for i in model.labels_]

print('############# center"')
print(model.cluster_centers_)

#on affiche les points avce les différentes couleurs des clusters
plt.scatter(x, y, c=colors)

plt.title('Keyword Issue Map')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('ScatterPlot_05.png')
plt.show()