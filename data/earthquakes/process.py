import math
import numpy as np
import pandas as pd
import dateutil.parser
from scipy.linalg import null_space
from sklearn.neighbors import BallTree

class EarthquakeGenerator:
    '''
    this class read the earthquake data, and create event sequences by rotating
    event around a chosen random center, the event features are the latitudes
    and the longitudes of the earthquakes
    '''

    def __init__():
        '''
        read earthquakes, return latitudes, logitiudes and timestamps for
        earthquke with magitude greater than 4.0
        '''
        dat = pd.read_csv("earthquakes.csv")
        leid = [i for i in range(len(dat)) if dat["mag"][i] > 4.0]

        self.lats = np.radians(np.array([dat["latitude"][i]  for i in leid]))
        self.lons = np.radians(np.array([dat["longitude"][i] for i in leid]))
        
        self.ts = np.array([dateutil.parser.parse(dat["time"][i]).timestamp() for i in leid])
        self.bt = BallTree(np.array([self.lats, self.lons]).T, metric="haversine")

    def sample_sphere():
        '''
        this function is used to sample a point on sphere
        and return its latitude and longitude in radius
        '''
        vec = np.random.randn(3)
        vec /= np.linalg.norm(vec)
    
        #----------------------------------------------
        lat = np.arcsin(vec[2])
        #----------------------------------------------
        if vec[0] > 0:
            lon = np.arctan(vec[1]/vec[0])
        elif vec[1] > 0:
            lon = np.arctan(vec[1]/vec[0])+math.pi
        else:
            lon = np.arctan(vec[1]/vec[0])-math.pi
        #----------------------------------------------
    
        return np.array([lat, lon])

    def ortho_axes(center_coord, rand_rot=False):
        '''
        this function return a 2-vector basis that is
        orthogonal to the input vector
        '''
        vec1 = np.cross([0,0,1], center_coord)
        assert np.linalg.norm(vec1) != 0.0
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = null_space([center_coord, vec1]).T[0]
        axes_ori = np.array([vec1, vec2]).T

        theta = np.radians(np.random.rand()*360.0) if rand_rot else 0
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c,-s], [s, c]])
    
        axes = np.dot(axes_xz, R)
    
        return axes

    def event_seqs(centers=None, num_seqs=10, seed0=False, radius=0.1, rand_rot=True, scale=1.0, time_cutoff=np.inf):
        if centers is not None:
            if rand_rot:
                print("warning: when centers are provided, the rand_rot is set to false")
            rand_rot=False
        else:
            if seed0:
                np.random.seed(0)
            centers = [sample_sphere() for i in range(num_seqs)]

        # now, the get the index set from center
        idxsets = bt.query_radius(centers, r=radius)

        # reference to stored values
        lats = self.lats
        lons = self.lons

        # create one event sequence for each center
        event_seqs = []

        for center, ids in zip(centers, idxsets):
            center_coord = np.array([np.sin(center[0]), np.cos(center[0])*np.cos(center[1]), 
                                                        np.cos(center[0])*np.sin(center[1])])
            event_coords = np.array([np.sin(lats[ids]), np.cos(lats[ids])*np.cos(lons[ids]), 
                                                        np.cos(lats[ids])*np.sin(lons[ids])]).T
            event_embeddings = np.dot(event_coords, ortho_axes(center_coord, rand_rot))
            event_times = self.ts[ids]

            event_seq = [(t*scale, k) for t,k in zip(event_times, event_embeddings) if t*scale < time_cutoff]

            event_seqs.append(event_seq)
 
        return event_seqs


if __name__ == "__main__":
    num_seqs = 10
    radius = 0.1
    eg = EarthquakeGenerator()
