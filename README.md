# Image_Feature_Extraction



This program extracts features for Caltech Dataset images and using these features we find K similar images to the input image.

Task 1:
1. To execute task 1 run the phase1_mongodb.py file
2. This program will compute below features and provide subsequent feature vectors
- Color Moments (10X10X3X3)
- HOG (10X10X9)
- Resnet - AvgPool (1024)
- Resnet - Layer3 (1024)
- Resnet - Fully Connected (1000)

Task 2:
1. To execute task 2 run the mongo_setup.py file
2. This file will save all feature vectors in mongodb database

Task 3:
1. To execute task 3 run the FindSimilarimages.py file
2. Provide index id of input image and value k to display K closest images to the input
3. The program then calculates distances and similarities of all images in dataset with reference to our input image
4. Output of closest images is determined by these distance/similarity
5. Output of k nearest images is then displayed to user

Requirements:
1. Python V3.7 + (Version 3.7 recommended)
2. MongoDB Shell (mongodb-windows-x86_64-7.0.1)
3. IDE supporting python files (VS Code, Jupyter) (Optional)
4. Python Libraries:
- torchvision
- matplotlib
- pymongo
- scipy




