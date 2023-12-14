#### Load and show football graph 


import urllib.request
import io
import zipfile

import matplotlib.pyplot as plt
import networkx as nx

#Link to the dataset
url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

sock = urllib.request.urlopen(url)  # open URL
s = io.BytesIO(sock.read())  # read into BytesIO "file"
sock.close()

zf = zipfile.ZipFile(s)  # zipfile object
txt = zf.read("football.txt").decode()  # read info file
gml = zf.read("football.gml").decode()  # read gml data
# throw away bogus first line with # from mejn files
gml = gml.split("\n")[1:]
G_foot = nx.parse_gml(gml)  # parse gml data

print(txt)
# print degree for each team - number of games
for n, d in G_foot.degree():
    print(f"{n:20} {d:2}")

options = {"node_color": "black", "node_size": 50, "linewidths": 0, "width": 0.1}

pos = nx.spring_layout(G_foot, seed=1969)  # Seed for reproducible layout
nx.draw(G_foot, pos, **options)
plt.show()

#### Initialize randomly with 12 spins/categories
init_G_football = random_initialization(G_foot,12)
