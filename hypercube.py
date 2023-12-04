import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import collections  as mc
from matplotlib import cm
import sys


def rotate(vertices : np.ndarray, theta : float, rot : np.ndarray, ndim : int) -> np.ndarray:
    """
    Rotate the hypercube in n-ndimensional space

    vertices: static vertices of hypercube
    theta: angle to rotate by
    rot: pair of indices representing axis to rotate around
    ndim: number of dimensions
    """
    # construct rotation matrix
    vals = np.array([np.cos(theta),np.sin(theta),-np.sin(theta),np.cos(theta)])
    rmtx = np.identity(ndim)
    for i, val in enumerate(vals):
        x = rot[i//2]
        y = rot[i%2]
        rmtx[x,y] = val

    # transform shape
    vec = np.transpose(rmtx @ vertices.T)
    return vec


def update(frame : int, ndim : int, dtheta : np.ndarray, rot : np.ndarray, disp : float, dvec : np.ndarray, vertices : np.ndarray, edges : np.ndarray, ecols : np.ndarray) -> None:
    '''
    Update the hypercube's rotation

    frame: frame index
    ndim: number of dimensions
    dtheta: change in rotation angle per tick along each dimension
    rot: list of paired indices representing axes to rotate around
    disp: displacement from origin (used to set axis limits)
    dvec: displacement vector
    vertices: static vertices of hypercube
    edges: list of pairs of vertex indices
    ecols: list of edge colors
    '''
    
    # rotate shape
    vtxs = vertices[:,:] # copy hypercube
    for i, r in enumerate(rot): # apply 2D rotations sequentially
        theta = frame * dtheta[i]
        vtxs = rotate(vtxs,theta,r,ndim)

    vec = vtxs[:,:] + dvec # apply offset

    # project to 2D
    while vec.shape[1] > 2:
        idx = vec.shape[1] -1
        delta = dvec[idx] / vec[:,idx]
        adjust = np.asarray([delta]).T @ np.ones((1,vec.shape[1]))
        vec *= adjust
        vec = vec[:,:idx]

    # reset axis
    ax.clear()
    space = disp
    ax.set_xlim(-space,space)
    ax.set_ylim(-space,space)
    ax.set_aspect('equal')

    # render
    ax.scatter(vec[:,0],vec[:,1],c='k',s=0.75) # draw vertices
    lines = [[tuple(vec[edges[i,0],:]),tuple(vec[edges[i,1],:])] for i in range(edges.shape[0])] #
    lc = mc.LineCollection(lines, colors=ecols, linewidths=0.5) 
    ax.add_collection(lc)


def get_cli_value(i : int, dft : int, min : int = 0) -> int:
    """
    Get integer value from CLI

    i: index of argument
    dft: default value
    """
    if len(sys.argv) > i:
        try:
            val = int(sys.argv[i])
        except:
            print('Argument {}: Invalid value; cannot convert to integer. Using default'.format(i))
            return dft
        if val < min:
            print('Argument {}: Value must be >= {}! Using default'.format(i,min))
            return dft
        return val
    else:
        return dft


if __name__ == "__main__":
    # --- Parameters ---
    ndim = 4 # (default) number of dimensions
    step = 1 # (default) separation between rotation axis pairs
    basespd = 0.2 # base rotation speed
    disp = 2 # displacement from origin

    # --- get inputs from CLI ---
    # if no CLI argument, uses default values set above
    ndim = get_cli_value(1,ndim,2)
    step = get_cli_value(2,step,1)


    # --- Construct Hypercube & Initialize Renderer ---
    # generate rotation information
    spd = np.round(basespd/ndim,2) # adjusted base rotation speed
    dtheta = np.array([spd * (i+1) for i in range(ndim-1)])
    rot = np.array([[i,i+step] for i in range(ndim-1)])
    rot[rot >= ndim] = 0

    # create displacement vector
    # this ensures that the hypercube isn't projected onto a plane that intersects it
    dvec = np.ones(ndim) * disp
    dvec[:2] = 0 # maintain center on origin in x,y

    # construct unit hypercube centered on origin
    nvtx = 2**ndim
    nedges = 2**(ndim-1) * ndim
    vertices = ((np.arange(nvtx)[:,None] & (1 << np.arange(ndim))) > 0) - 0.5

    # find edges
    axcols = cm.rainbow(np.linspace(0, 1, ndim))
    ecols = [] # colors of lines (color-coded by axis)
    edges = [] # indices of vertices on each edge
    for i in range(ndim):
        vtx = list(range(nvtx)) # list of remaining vertices
        while(len(vtx) > 0):
            v1 = vtx.pop(0)
            valid = vtx.copy()
            # dismiss vertices that don't share an axis
            for j in range(ndim):
                if i == j : valid = [k for k in valid if vertices[v1,j] != vertices[k,j]]
                else : valid = [k for k in valid if vertices[v1,j] == vertices[k,j]]
            v2 = valid[0]
            vtx.remove(v2)
            edges.append([v1,v2])
            ecols.append(axcols[i])
    edges = np.asarray(edges)
    ecols = np.asarray(ecols)

    # apply initial rotation
    '''
    itheta = 0 # initial rotation
    irot = [1,2] # initial rotation axes
    vertices = rotate(vertices,itheta,irot,ndim)
    '''


    # --- Render ---
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, update, interval=50, fargs=(ndim,dtheta,rot,disp,dvec,vertices,edges,ecols))
    plt.show()
    exit(0)
