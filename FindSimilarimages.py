
from Image_Feature_Extraction.phase1_mongodb import loadDataset
from pymongo import MongoClient
import matplotlib.pyplot as plt
import scipy
import math
import numpy as np
import torchvision.transforms

client = MongoClient()
client = MongoClient(host="localhost", port=27017)

caltectDataset = loadDataset()

# Input image ID
image_id = int(input("Enter Image ID of the image: "))

# K similar images to be found
k = int(input("Enter value of k: "))

db = client.Multimedia_Web_DBs

feature_descriptors = list(db.Feature_Descriptors.find({}))
n = len(feature_descriptors)

euclidean_distances = []
cosine_distances = []
pearson_correlation = []
layer3_pearson_correlation = []
avgpool_pearson_correlation = []

# Calculate distance/similarity for each feature by retrieving data from database
for i in range(n):
  if i == image_id:
    continue
  
  if feature_descriptors[i].get("color_moments") and len(feature_descriptors[image_id])>2:
    sim = math.dist(feature_descriptors[image_id]["color_moments"], feature_descriptors[i]["color_moments"])
    if not math.isnan(sim):
      euclidean_distances.append({"_id": feature_descriptors[i]["_id"],  "similarity": math.dist(feature_descriptors[image_id]["color_moments"], feature_descriptors[i]["color_moments"])})

  if feature_descriptors[i].get("hog"):
    cosine_distances.append({"_id": feature_descriptors[i]["_id"],  "similarity": (np.dot(feature_descriptors[image_id]["hog"], feature_descriptors[i]["hog"]) / (np.linalg.norm(feature_descriptors[image_id]["hog"]) * np.linalg.norm(feature_descriptors[i]["hog"])))})

  if feature_descriptors[i].get("avgpool") and len(feature_descriptors[image_id])>2:
    avgpool_pearson_correlation.append({"_id": feature_descriptors[i]["_id"], "similarity": scipy.stats.pearsonr(feature_descriptors[image_id]["avgpool"], feature_descriptors[i]["avgpool"]).statistic})

  if feature_descriptors[i].get("layer3") and len(feature_descriptors[image_id])>2:
    layer3_pearson_correlation.append({"_id": feature_descriptors[i]["_id"], "similarity": scipy.stats.pearsonr(feature_descriptors[image_id]["layer3"], feature_descriptors[i]["layer3"]).statistic})

  if feature_descriptors[i].get("fc") and len(feature_descriptors[image_id])>2:
    pearson_correlation.append({"_id": feature_descriptors[i]["_id"], "similarity": scipy.stats.pearsonr(feature_descriptors[image_id]["fc"], feature_descriptors[i]["fc"]).statistic})

# Sorting distances and similarities to get k top images
euclidean_similarity = sorted(euclidean_distances, key=lambda x: x["similarity"], reverse=False)
cosine_similarity = sorted(cosine_distances, key=lambda x: x["similarity"], reverse=True)
avgpool_pearson_similarity = sorted(avgpool_pearson_correlation, key=lambda x: x["similarity"], reverse=True)
layer3_pearson_similarity = sorted(layer3_pearson_correlation, key=lambda x: x["similarity"], reverse=True)
pearson_similarity = sorted(pearson_correlation, key=lambda x: x["similarity"], reverse=True)

# Data to be displayed
figure_data = [
  euclidean_similarity,
  cosine_similarity,
  avgpool_pearson_similarity,
  layer3_pearson_similarity,
  pearson_similarity,
]

# Creating grid to display all images
num_rows = 6
fig, axes = plt.subplots(num_rows, k, figsize=(10,10))  # Adjust the figsize as needed

for i in range(6):
    for j in range(k):
        
        if i == 0:
          if j == 0:
            image,_ =  caltectDataset[image_id]
            image = torchvision.transforms.Resize((300,300))(image)
            axes[i, j].imshow(image.permute(1, 2, 0))
            axes[i,j].set_title("Input image")
          axes[i, j].axis('off')
          continue
        if len(figure_data[i-1])>0 :
          match = caltectDataset[figure_data[i - 1][j]["_id"]]
          match = torchvision.transforms.Resize((300,300))(match[0])
          axes[i, j].imshow(match.permute(1, 2, 0))
          axes[i, j].axis('off')
          axes[i, j].set_title(f"Image ID: ({figure_data[i - 1][j]['_id']})\nScore : {round(figure_data[i - 1][j]['similarity'], 5)}")

# Adjust spacing between subplots
plt.tight_layout(pad = 2.0)
plt.show()





