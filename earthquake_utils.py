import math
import numpy as np
import pandas as pd
import dateutil.parser
from scipy.linalg import null_space
from sklearn.neighbors import BallTree

# this class read the earthquake data, and create event sequences by rotating
# event around a chosen random center, the event features are the latitudes
# and the longitudes of the earthquakes
class EarthquakeGenerator:

    # read earthquakes, return latitudes, logitiudes and timestamps for
    # earthquke with magitude greater than 4.0
    def __init__(self):
        dat = pd.read_csv("data/earthquakes/earthquakes.csv")
        leid = np.where(np.array(dat["mag"] > 4.0))[0]

        self.lats = np.radians(np.array(dat["latitude"][leid]))
        self.lons = np.radians(np.array(dat["longitude"][leid]))
        
        self.ts = np.array([dateutil.parser.parse(dat["time"][i]).timestamp() for i in leid])
        self.bt = BallTree(np.array([self.lats, self.lons]).T, metric="haversine")

    # this function is used to sample a point on sphere
    # and return its latitude and longitude in radius
    @staticmethod
    def sample_sphere():
        vec = np.random.randn(3)
        vec /= np.linalg.norm(vec)
    
        lat = np.arcsin(vec[2])
        if vec[0] > 0:
            lon = np.arctan(vec[1]/vec[0])
        elif vec[1] > 0:
            lon = np.arctan(vec[1]/vec[0])+math.pi
        else:
            lon = np.arctan(vec[1]/vec[0])-math.pi

        return np.array([lat, lon])

    # this function return a 2-vector basis that is
    # orthogonal to the input vector
    @staticmethod
    def ortho_axes(center_coord, rand_rot=False):
        vec1 = np.cross([0, 0, 1], center_coord)
        assert np.linalg.norm(vec1) != 0.0
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = null_space([center_coord, vec1]).T[0]
        axes_xz = np.array([vec1, vec2]).T

        theta = np.radians(np.random.rand()*360.0) if rand_rot else 0
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
    
        axes = np.dot(axes_xz, rot)
    
        return axes

    def event_seqs(self, centers=None, num_seqs=10, seed0=False, radius=0.1, rand_rot=True, scale=1.0, time_cutoff=np.inf):
        if centers is not None:
            if rand_rot:
                print("warning: when centers are provided, the rand_rot is set to false")
            rand_rot = False
        else:
            if seed0:
                np.random.seed(0)
            centers = [self.sample_sphere() for _ in range(num_seqs)]

        # now, the get the index set from center
        idxsets = self.bt.query_radius(centers, r=radius)

        # reference to stored values
        lats = np.array(self.lats)
        lons = np.array(self.lons)

        # create one event sequence for each center
        event_seqs = []
        for center, ids in zip(centers, idxsets):
            center_coord = np.array([np.cos(center[0])*np.cos(center[1]), np.cos(center[0])*np.sin(center[1]), np.sin(center[0])])
            event_coords = np.array([np.cos(lats[ids])*np.cos(lons[ids]), np.cos(lats[ids])*np.sin(lons[ids]), np.sin(lats[ids])]).T - center_coord
            event_embeddings = np.dot(event_coords, self.ortho_axes(center_coord, rand_rot))
            event_times = self.ts[ids]

            event_seq = [(t*scale, list(k)) for t, k in zip(event_times, event_embeddings) if t*scale < time_cutoff]

            event_seqs.append(sorted(event_seq))
 
        return event_seqs


if __name__ == "__main__":
    eg = EarthquakeGenerator()
    event_seqs = eg.event_seqs(centers=[np.radians([37.229564, -120.047533])], num_seqs=10, radius=0.2, scale=1.0)
