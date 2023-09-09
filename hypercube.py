import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import collections  as mc
from matplotlib import cm


def rotate(coords : np.ndarray, theta : float, rot : list[int], dim : int) -> np.ndarray:
    # construct rotation matrix
    vals = [np.cos(theta),np.sin(theta),-np.sin(theta),np.cos(theta)]
    rmtx = np.identity(dim)
    i = -1
    for x in rot:
        for y in rot:
            i += 1
            rmtx[x,y] = vals[i]

    # transform shape
    vec = np.transpose(rmtx @ coords.T)
    return vec


dim = 5 # number of dimensions
step = 3
#rot = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]#[[0,2],[2,3]] # axes to rotate
rot = [[0,1],[2,3]] # axes to rotate
disp = 3 # displacement from origin
dtheta = [0.02,0.04,0.06] # delta theta
itheta = 0#np.pi / 16 # initial rotation
irot = [0,1] # initial rotation axes

# generate rotation information
rot = np.asarray([[i,i+step] for i in range(dim-1)])
rot[rot >= dim] = 0
dtheta = [0.03 * i for i in range(dim-1)]

# create displacement vector
dvec = np.ones(dim) * disp
dvec[:2] = 0

# construct "unit" hypercube (nvtx * dim)
nvtx = 2**dim
nedges = 2**(dim-1)*dim
vertices = 2 * ((np.arange(2**dim)[:,None] & (1 << np.arange(dim))) > 0) - 1

# find edges
axcols = cm.rainbow(np.linspace(0, 1, dim))
#axcols = [[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,1],[1,0,1,1],[0,1,1,1],[1,1,0,1]]
ecols = [] # colors of lines (color-coded by axis)
edges = []#np.zeros((nedges,2)) # indices of vertices on each edge
for i in range(dim):
    vtx = list(range(nvtx)) # list of remaining vertices
    while(len(vtx) > 0):
        v1 = vtx.pop(0)
        valid = vtx.copy()
        # dismiss verties that don't share an axis
        for j in range(dim):
            if i == j : valid = [k for k in valid if vertices[v1,j] != vertices[k,j]]
            else : valid = [k for k in valid if vertices[v1,j] == vertices[k,j]]
        v2 = valid[0]
        vtx.remove(v2)
        edges.append([v1,v2])
        ecols.append(axcols[i])
edges = np.asarray(edges)
ecols = np.asarray(ecols)

# apply initial rotation
vertices = rotate(vertices,itheta,irot,dim)


# render
fig, ax = plt.subplots()

def update(frame : int, dtheta : list[float], dim : int, disp : float, rot : list[list[int]], dvec : np.ndarray, coords : np.ndarray, edges : np.ndarray, ecols : np.ndarray) -> None:
    '''
    update the ND shape render; display at new rotation
    '''
    
    # rotate shape
    tcoords = coords[:,:]
    for i, r in enumerate(rot):
        theta = frame * dtheta[i]
        tcoords = rotate(tcoords,theta,r,dim)
    vec = tcoords[:,:] + dvec

    '''# construct rotation matrix
    theta = frame * dtheta
    vals = [np.cos(theta),np.sin(theta),-np.sin(theta),np.cos(theta)]
    rmtx = np.identity(dim)
    i = -1
    for x in rot:
        for y in rot:
            i += 1
            rmtx[x,y] = vals[i]

    # transform shape
    tcoords = np.transpose(rmtx @ coords.T) + dvec
    vec = tcoords[:,:]'''

    # project to 2D
    while vec.shape[1] > 2:
        idx = vec.shape[1] -1
        delta = dvec[idx] / vec[:,idx]
        adjust = np.asarray([delta]).T @ np.ones((1,vec.shape[1]))
        #print(adjust)
        #print(vec)
        vec = vec * adjust
        vec = vec[:,:idx]

    # reset axis
    ax.clear()
    space = disp * 2
    ax.set_xlim(-space,space)
    ax.set_ylim(-space,space)
    ax.set_aspect('equal')

    # render
    ax.scatter(vec[:,0],vec[:,1],c='k',s=0.75) # draw vertices

    lines = [[tuple(vec[edges[i,0],:]),tuple(vec[edges[i,1],:])] for i in range(edges.shape[0])]
    '''lcols = []
    for i in range(edges.shape[0]):
        z1 = tcoords[edges[i,0],:]
        z2 = tcoords[edges[i,1],:]
        r = min(abs(z1[rot[1]]) / disp,1)
        g = min(abs(z2[rot[1]]) / disp,1)
        b = 0
        a = 1
        lcols.append([r,g,b,a])
    lcols = np.asarray(lcols)'''
    #lcols = np.asarray([(1,1,1,tcoords[edges[i]]) for i in range(edges.shape[0])])
    lc = mc.LineCollection(lines, colors=ecols, linewidths=0.5)
    ax.add_collection(lc)


ani = animation.FuncAnimation(fig, update, interval=50, fargs=(dtheta,dim,disp,rot,dvec,vertices,edges,ecols))
plt.show()
exit(0)
