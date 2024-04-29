import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure, morphology
from tabulate import tabulate
import utils_primitives as up


def main() -> None:
    pass
    
# sphere surface area
shape = (100, 100, 100)
slice_ix = shape[0] // 2
radiuses = [10, 25, 45]

save_report = True
save_primitives = False

areas_formulas = []
areas_march_cubes = []
areas_diffs = []

volume_formulas = []
volume_count = []
volume_diffs = []

for radius in radiuses:
    sphere_image = up.create_sphere(radius, shape)
    verts, faces, normals, values = measure.marching_cubes(sphere_image) # type: ignore    
    surf_area = round(measure.mesh_surface_area(verts, faces),2)
    formula_area = round(4 * np.pi * radius**2, 2)
    difference_relative2 = np.round((surf_area - formula_area)/formula_area * 100, 2)
    
    formula_volume = round(4/3 * np.pi * radius**3, 2)
    volume = np.count_nonzero(sphere_image)
    vol_difference_relative2 = np.round((volume - formula_volume)/formula_volume * 100, 2)
    
    areas_formulas.append(formula_area)
    areas_march_cubes.append(surf_area)
    areas_diffs.append(difference_relative2)
    volume_formulas.append(formula_volume)
    volume_count.append(volume)
    volume_diffs.append(vol_difference_relative2)    
        
    #up.show_image_projections(sphere_image) #only for Jupyter notebook
    if save_primitives:
        up.store_primitive(sphere_image, "sphere_radius_" + str(radius).zfill(3) + ".nii")

print("Sphere primitives processed")
if save_report:
    df_sphere = up.create_dataframe("sphere", radiuses, None, areas_formulas, areas_march_cubes, areas_diffs, volume_formulas, volume_count, volume_diffs)
    print("\n" + tabulate(df_sphere, headers='keys', tablefmt='psql', showindex=True)) 

# cylinder
shape = (100, 100, 100)
radiuses = [10, 25, 45]
heights = [40, 60, 80]

areas_formulas = []
areas_march_cubes = []
areas_diffs = []

volume_formulas = []
volume_count = []
volume_diffs = []


for radius, height in zip(radiuses, heights):
    cylinder_image = up.create_cylinder(radius, height, shape)
    formula_area = 2 * np.pi * radius * height + 2 * np.pi* radius**2    
    verts, faces, normals, values = measure.marching_cubes(cylinder_image) # type: ignore    
    surf_area = measure.mesh_surface_area(verts, faces)
    difference_relative2 = np.round((surf_area - formula_area)/formula_area * 100, 2)

    formula_volume = round(np.pi * radius**2 * height, 2)
    volume = np.count_nonzero(cylinder_image) 
    vol_difference_relative2 = np.round((volume - formula_volume)/formula_volume * 100, 2)

    if save_report:
        areas_formulas.append(formula_area)
        areas_march_cubes.append(surf_area)
        areas_diffs.append(difference_relative2)
        volume_formulas.append(formula_volume)
        volume_count.append(volume)
        volume_diffs.append(vol_difference_relative2)


    #up.show_image_projections(cylinder_image) #only for Jupyter notebook

    if save_primitives:
        up.store_primitive(cylinder_image, "cylinder_radius_" + str(radius).zfill(3) + "_height_" + str(height).zfill(3) + ".nii")
        

print("Cylinder primitives processed")
if save_report: 
    df_cylinder = up.create_dataframe("cylinder", radiuses, heights, areas_formulas, areas_march_cubes, areas_diffs, volume_formulas, volume_count, volume_diffs)
    print("\n" + tabulate(df_cylinder, headers='keys', tablefmt='psql', showindex=True)) 

shape = (100, 100, 100)
radiuses = [10, 25, 45]
heights = [40, 60, 80]
extra_space = 20

areas_formulas = []
areas_march_cubes = []
areas_diffs = []

volume_formulas = []
volume_count = []
volume_diffs = []

for radius, height in zip(radiuses, heights):
    cylindroid_image = up.create_cylindroid(radius, height, extra_space, shape)
    formula_area = 2 * np.pi * radius * height + 4 * np.pi * radius**2    
    verts, faces, normals, values = measure.marching_cubes(cylindroid_image) # type: ignore    
    surf_area = measure.mesh_surface_area(verts, faces)
    difference_relative2 = np.round((surf_area - formula_area)/formula_area * 100, 2)
    formula_volume = round(4/3 * np.pi * radius**3 + np.pi * radius**2 * height , 2)
    volume = np.count_nonzero(cylindroid_image)
    vol_difference_relative2 = np.round((volume - formula_volume)/formula_volume * 100, 2)
    
    if save_report:
        areas_formulas.append(formula_area)
        areas_march_cubes.append(surf_area)
        areas_diffs.append(difference_relative2)
        volume_formulas.append(formula_volume)
        volume_count.append(volume)
        volume_diffs.append(vol_difference_relative2)
        
    #up.show_image_projections(cylindroid_image) #only for Jupyter notebook
    if save_primitives:
        up.store_primitive(cylindroid_image, "cylindroid_radius_" + str(radius).zfill(3) + "_central_height_" + str(height).zfill(3) + ".nii")
print("Cylindroid primitives processed")

if save_report:
    df_cylindroid = up.create_dataframe("cylindroid", radiuses, heights, areas_formulas, areas_march_cubes, areas_diffs, volume_formulas, volume_count, volume_diffs)
    print("\n" + tabulate(df_cylindroid, headers='keys', tablefmt='psql', showindex=True)) 
    df = pd.concat([df_sphere, df_cylinder, df_cylindroid], axis = 0,  ignore_index = True)
    up.save_tables(df, "primitives_benchmark_report")


if __name__ == '__main__': 
    main()