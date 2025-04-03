import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    
    # Assign the color to the player using KMeans Cluster within the bounding box
    def get_player_color(self, frame, bbox):
        
        # Preprocess the image for training on KMeans cluster model
        img = cv.imread(frame[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        top_half_img = img[:height//2, width]
        top_half_img = top_half_img.reshape(-1, 3)
        
        # KMeans cluster Training
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=42)
        kmeans.fit(top_half_img)
        
        # Reshape gray scale image which consist of two cluster i.e., player and background
        labels = kmeans.labels_
        clustered_img = labels.reshape(top_half_img.shape(0), top_half_img.shape(1))
        
        # Assign the cluster to the player and background 
        corner_labels = [clustered_img[0, 0], clustered_img[0, -1], clustered_img[-1, 0], clustered_img[-1, 1]]
        background_cluster = max(corner_labels, key=corner_labels.count)
        player_cluster = 1 - background_cluster
        
        # Get the player color
        color = kmeans.cluster_centers_[player_cluster]
        return color
    
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detections in player_detections.items():
            bbox = player_detections['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(player_colors)
        
        self.kmeans = kmeans
        
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        
    def get_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, bbox)
        
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        
        self.player_team_dict[player_id] = team_id
        
        return team_id