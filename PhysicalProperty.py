'''
Created on 2020/01/12

Physical properties
0.Steel, 1.Stainless steel, 2.Aluminium, 3.Acrylic resin, 4.Water

'''
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors


# velocity of compressional waves [m/s]
vp = np.array((5900, 5780, 6260, 2730, 1483), dtype = 'float')
# velocity of shear waves [m/s]
vs = np.array((3230, 3060, 3080, 1430, 0), dtype = 'float')
# density [kg/m3]
density = np.array((7870, 7800, 2700, 1180, 1000), dtype = 'float')
# colormap
cmap = cm.get_cmap("jet")
cmap_data1 = cmap(np.arange(cmap.N))
cmap_data2 = cmap(np.arange(cmap.N))
cmap_data3 = cmap(np.arange(cmap.N))
cmap_data1[:, 3] = 1.00
jet100 = colors.ListedColormap(cmap_data1)
cmap_data2[:, 3] = 0.75
jet050 = colors.ListedColormap(cmap_data2)
cmap_data3[:, 3] = 0.50
jet010 = colors.ListedColormap(cmap_data3)
cmap = [jet100, jet100, jet100, jet050, jet010]