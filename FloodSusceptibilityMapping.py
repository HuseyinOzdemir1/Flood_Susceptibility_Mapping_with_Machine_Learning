import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,ConfusionMatrixDisplay
import rioxarray

# Flooded and non-flooded points to be used in the training and testing phase
pointData = gpd.read_file('sampledPoint.shp')

# Raster data of the parameters required for the ML algorithm
demRaster = rasterio.open('Dem.tif')
slopeRaster = rasterio.open('SlopeDem.tif')
aspectRaster = rasterio.open('AspectDem.tif')
curvatureRaster = rasterio.open('CurvatureDem.tif')
rainfallRaster = rasterio.open('rainfall.tif')

"""Since the DEM data covers a very large area, the area where the ML algorithm will be estimated has been reduced."""
sampledemRaster = rasterio.open('SampleDem.tif')

# x and y coordinates of the point data
coords = []

for point in pointData['geometry']:
    
    x = point.xy[0][0]
    y = point.xy[1][0]

    coords.append((x,y))

del x,y

# We can plot the DEM and points with the help of rasterio.plot module 
fig, ax = plt.subplots(figsize=(15,15))
pointData.plot(ax = ax, color = 'orangered')
show(demRaster, ax = ax)

del fig, ax

""" The parameter values of each point can be stored in a list. """
sampleElevation = []
sampleSlope = []
sampleAspect = []
sampleCurvature = []
sampleRainfall = []

frame = {"Elevation": [demRaster, sampleElevation],
         "Slope": [slopeRaster, sampleSlope],
         "Aspect": [aspectRaster, sampleAspect],
         "Curvature": [curvatureRaster, sampleCurvature],
         "Rainfall": [rainfallRaster, sampleRainfall]}

for item in frame.keys():
    
    RasterValue = [x for x in frame[item][0].sample(coords)]

    for value in RasterValue:

        frame[item][1].append(value[0])  

del item, value, frame 

# And then, we can create a data frame with class and parameters
frame = {"Class": list(pointData["Class"]),"Elevation": sampleElevation,"Slope": sampleSlope,
         "Aspect": sampleAspect, "Curvature": sampleCurvature, "Rainfall": sampleRainfall}
df = pd.DataFrame(frame)

# Droppimg the nan values 
df = df.dropna()

# Distinguishing the parameters and prediction class
X=df.drop('Class',axis=1)
y=df['Class'].copy()

del frame 

# Plotting distribution of parameters
g = sns.PairGrid(df, hue="Class", corner=True)
g.map_lower(sns.kdeplot, hue=None, levels=5, color=".2")
g.map_lower(sns.scatterplot, marker="+")
g.map_diag(sns.histplot, element="step", linewidth=0, kde=True)
g.add_legend(frameon=True)
g.legend.set_bbox_to_anchor((.61, .6))

del g

# Splitting the train and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,stratify=y)

# Creating the GradientBoostingClassifier model
model = GradientBoostingClassifier(learning_rate=0.08, min_samples_leaf= 4, min_samples_split=2)
model.fit(X_train, y_train)

# Prediction of the train and test set
y_trainpred=model.predict(X_train)
y_testpred=model.predict(X_test)

# Performance of the model for 3 different metrics
print('Train ROC score: ',roc_auc_score(y_train,y_trainpred))
print('Test ROC score: ',roc_auc_score(y_test,y_testpred))
print('Train F1 score: ',f1_score(y_train,y_trainpred))
print('Test F1 score: ',f1_score(y_test,y_testpred))
print('Train Accuracy score: ',accuracy_score(y_train,y_trainpred))
print('Test Accuracy score: ',accuracy_score(y_test,y_testpred))


# Plotting the confusion matrix of train and test set
ConfusionMatrixDisplay.from_estimator(model,
                      X_train,
                      y_train,
                      values_format='d',
                      display_labels=['No Flood Risk','Flood Risk']).figure_.set_size_inches(5,5)

ConfusionMatrixDisplay.from_estimator(model,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=['No Flood Risk','Flood Risk']).figure_.set_size_inches(5,5)

""" Finding the coordinates of each pixel in the scope of area (In this work, sampledemRaster is the area of interest) """
band1 = sampledemRaster.read(1)
height = band1.shape[0]
width = band1.shape[1]
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
xs, ys = rasterio.transform.xy(sampledemRaster.transform, rows, cols)
lons= np.array(xs)
lats = np.array(ys)

del xs, ys, band1, height, width

# Generating a function to convert Numpy array to list 
def coordstoList(coords):

    coordsList = []
    for coordarray in coords:
        for coord in coordarray:

            coordsList.append(coord)
            
    return coordsList

xcoords = coordstoList(lons)
ycoords = coordstoList(lats)

del lons, lats

# A list that contain the x,y coordinates as a tuple

coordsList = []

for index in range(len(xcoords)):
    coord = (xcoords[index], ycoords[index])
    coordsList.append(coord)

""" The parameter values of each pixel can be stored in a list. """
sampleElevation = []
sampleSlope = []
sampleAspect = []
sampleCurvature = []
sampleRainfall = []

frame = {"Elevation": [demRaster, sampleElevation],
         "Slope": [slopeRaster, sampleSlope],
         "Aspect": [aspectRaster, sampleAspect],
         "Curvature": [curvatureRaster, sampleCurvature],
         "Rainfall": [rainfallRaster, sampleRainfall]}

for item in frame.keys():
    
    RasterValue = [x for x in frame[item][0].sample(coordsList)]

    for value in RasterValue:

        frame[item][1].append(value[0])  

# We can create a data frame with parameters, longitude and latitude
frame = {"Longitude": xcoords, "Latitude": ycoords, "Elevation": sampleElevation, "Slope": sampleSlope,
         "Aspect": sampleAspect, "Curvature": sampleCurvature, "Rainfall": sampleRainfall}
df = pd.DataFrame(frame)

# Dropping the nan values
df = df.dropna()

df1=df.drop('Longitude',axis=1)
X=df1.drop('Latitude',axis=1)

# Class probability of prediction of each pixel
yPrediction = model.predict_proba(X)

# Adding the flood probability to data frame
df["FSV"] = yPrediction[:,1]

# Transfering the predicted points to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y),crs="EPSG:4326")

gdf["x"] = gdf.x
gdf["y"] = gdf.y

# Points to raster with the help of rioxarray module
da = (df.set_index(["y","x"]).FSV.to_xarray())
da.rio.to_raster("FloodSusceptibilityMap.tif")

fsmRaster = rasterio.open("FloodSusceptibilityMap.tif")

# Plotting the flood susceptibility map
fig, ax = plt.subplots(figsize=(15,15))
plotRaster = show(fsmRaster, ax = ax, title= "Flood Susceptibility Map", cmap = 'RdYlGn_r')

