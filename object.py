class Object:
    def __init__(self, id, density, centroid):
        self.id = id
        self.densities = [density]
        self.centroids = [centroid]

    def get_latest_centroid(self):
        return self.centroids[-1]