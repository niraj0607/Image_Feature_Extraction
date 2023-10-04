from phase1_mongodb import *
from pymongo import MongoClient
import time

start_time = time.time()

client = MongoClient()
client = MongoClient(host="localhost", port=27017)

# Load Caltech Dataset
caltectDataset = loadDataset()

db = client.Multimedia_Web_DBs

# Connect the Database
feature_descriptors = db.Feature_Descriptors

n = len(caltectDataset)

# Create set of data to store in database 
for image_id in range(n):
  _, label = caltectDataset[image_id]
  if not checkChannel(caltectDataset[image_id]):
    data = {
      "_id": image_id,
      "color_moments": compute_color_moments(caltectDataset[image_id]),
      "hog": compute_hog(caltectDataset[image_id]),
      "avgpool": resnet50_avgpool(caltectDataset[image_id]),
      "layer3": resnet50_layer3(caltectDataset[image_id]),
      "fc": resnet50_fc(caltectDataset[image_id]),
      "label": label
    }
  # Handle Grayscale images
  else:
    data = {
      "_id": image_id,
      "hog": compute_hog(caltectDataset[image_id])
    }

# Store all features in database
  if feature_descriptors.find_one({"_id": image_id}):
    feature_descriptors.update_one({"_id": image_id}, {"$set": data})
    print(f"Updated feature descriptors for Image {image_id}")
  else:
    feature_descriptors.insert_one(data)
    print(f"Inserted feature descriptors for Image {image_id}")