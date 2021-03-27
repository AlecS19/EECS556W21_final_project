import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import PIL.Image as Image


#Load image
fileName = '260058'
img = Image.open('/home/alecsoc/Desktop/eecs556/EECS556W21_final_project/test_images/'+ fileName+'.jpg')
img_data = np.asarray(img)


n,m,c = img_data.shape
img_data_flat = np.reshape( img_data, (n*m,c) )
numSegs = 2

#Set up Gaussian Mixture Model
classif = GaussianMixture(
    n_components=numSegs,
    init_params='kmeans',
    covariance_type='full',
    tol=1e-4,
    max_iter=10)
classif.fit(img_data_flat)

classified = np.reshape( classif.fit_predict(img_data_flat) , (n,m) )
print(classified)
results = Image.fromarray( (classified * 255/(numSegs-1)).astype(np.uint8) )
results.save('/home/alecsoc/Desktop/eecs556/EECS556W21_final_project/test_images/'+ fileName+'_gmm'+'.jpg')

plt.imshow(classified)
plt.show()