import numpy as np
import nibabel as nib
from skimage.draw import disk
import pandas as pd
import matplotlib.pyplot as plt

# Showing projections of 3D image in Jupyter Notebook
def show_image_projections(im, figsize = (15,5) ):      
    
    cut_plane = im.shape[0] // 2    
    plt.figure(figsize=figsize)

    plt.subplot(1,3,1)
    plt.axis('off')
    plt.imshow(im[cut_plane, :, :], cmap='gray')
    plt.title('image1')
    
    cut_plane = im.shape[1] // 2
    plt.subplot(1,3,2)
    plt.axis('off')
    plt.imshow(im[:, cut_plane, :], cmap='gray')
    plt.title('image2')
    
    cut_plane = im.shape[2] // 2
    plt.subplot(1,3,3)
    plt.axis('off')
    plt.imshow(im[:, :, cut_plane], cmap='gray')
    plt.title('image3')    

    plt.tight_layout()
    plt.show()

'''
To create a sphere in a 3D numpy array with a given radius, you can use the equation of a sphere:
x^2 + y^2 + z^2 = r^2
Where x, y and z are the coordinates of each voxel in the array, and r is the radius of the sphere. 
You iterate over each voxel in the array, calculate its distance from the center of the sphere, and if the distance is less than or equal to the radius, 
you set the value of that voxel to 1, indicating that it is part of the sphere

Example usage:
radius = 5
shape = (20, 20, 20)
sphere = create_sphere(radius, shape)

This function create_sphere takes the radius of the sphere and the shape of the numpy array as input and returns a 3D numpy array 
representing the sphere with the specified radius. Note that this implementation is straightforward but not very efficient, especially for larger arrays and radii. 
Depending on your specific needs, you might want to explore more efficient algorithms like the midpoint circle algorithm adapted to 3D space.
'''
def create_sphere(radius, shape):
    center = np.array(shape) // 2
    grid = np.zeros(shape, dtype=np.uint8)
    
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if distance <= radius:
                    grid[x, y, z] = 1
                    
    return grid

'''
Function to create a cylinder, centered in the array of given shape
If centering is False, the span of the third dimension = height of cylinder
'''
def create_cylinder(radius, height, shape, centering = True):
    center = np.array(shape) // 2
    plane = np.zeros(shape[:2], dtype = np.uint8)
    rr, cc = disk(center[:2], radius, shape=shape[:2]) # type: ignore
    plane[rr, cc] = 1    
    cylinder_image = np.stack([plane]*height, 2)
    if centering:
        whole_image = np.zeros(shape, dtype = np.uint8)
        start = (shape[2] - height)//2
        whole_image[..., start : start+height] = cylinder_image
        return whole_image
    return cylinder_image

'''
Function to create a cylindrycal object consisting of cylinder and semispheres on tops.
The surface area of such figure is the surface area of the cylinder without top parts(2πrh) and surface area of the sphere, 
added to tops.
'''
def create_cylindroid(radius, height, extra_space, shape):
    shape4sphere =  shape[:2] + (radius*2 + extra_space,)  
    sphere_image = create_sphere(radius, shape4sphere)

    #splitting the sphere
    sphere_image1, sphere_image2 = sphere_image[..., : sphere_image.shape[-1]//2], sphere_image[..., sphere_image.shape[-1]//2 :]
    cylinder_image = create_cylinder(radius, height, shape, centering = False)
    cylindroid_image = np.concatenate([sphere_image1, cylinder_image, sphere_image2], 2)
    return cylindroid_image


'''
Function to store primitive in numpy array as nifty file
'''
def store_primitive(array, name):
    img_nifty = nib.Nifti1Image(array, affine=np.eye(4))
    nib.save(img_nifty, name)

'''
Function to create Pandas dataframe from collected data
'''

def create_dataframe(name, radiuses, heights, areas_formulas, areas_march_cubes, areas_diffs, volume_formulas, volume_count, volume_diffs):
    data = {"name": name,
            "radius" : radiuses,
            "height" : heights, 
            "surface by formula" : areas_formulas,
            "surface by algorithm" : areas_march_cubes,
            "difference area, %" : areas_diffs,
            "volume by formula": volume_formulas,
            "volume count" : volume_count,
            "difference volume, %" : volume_diffs
           }
    df = pd.DataFrame(data)
    return df

'''
Function to store report tables in csv and exel files
'''
def save_tables(df, name):
    df.to_csv(name + ".csv")
    df.to_excel(name + ".xlsx")    