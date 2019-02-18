# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 18:08:35 2019

@author: energ
"""

'''****************************************************************************
* Import Libraries
****************************************************************************'''

import colour
import tkinter         as tk
import multiprocessing as mp
import numpy           as np
import pylab           as plt
plt.ioff()

from colour.plotting import diagrams   as d
from   matplotlib    import colors     as cs
from   tkinter       import filedialog
from   PIL           import Image

import PIL.ExifTags
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150



'''****************************************************************************
* Set Runtime VAriables
****************************************************************************'''

if __name__ == '__main__':
    Temperature = 6500.0 #K  --must be a float!
    scale       = 1/10    #Image resize parm
    timeout     = 15      #seconds
    vmin,vmax   = 400,650 #nm
    cmap        = plt.cm.nipy_spectral


'''****************************************************************************
* Helper Functions
****************************************************************************'''

def get_file_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

#Calculate the dominant wavelength
def dom (row,xy_n) :
    return np.absolute(colour.dominant_wavelength(row,xy_n,reverse=False)[0])

#Calculate the complimentary wavelength
def com (row,xy_n) :
    return np.absolute(colour.dominant_wavelength(row,xy_n,reverse=True)[0])

#Calculate both dominant and complimentary wavelengths, in parallel
def domcom_spectra(xy,xy_n):
  with mp.Pool(processes=mp.cpu_count() - 1) as pool:
      dom_procs,com_procs = [],[]

      for i,row in enumerate(xy):
          dom_procs.append( pool.apply_async( dom,(row,xy_n) ) )
          com_procs.append( pool.apply_async( com,(row,xy_n) ) )

      r,c,d                   = xy.shape
      spectra_dom,spectra_com = np.zeros([2,r,c])
      for i,proc in enumerate(dom_procs):
          spectra_dom[i] =         proc.get(timeout=timeout)
          spectra_com[i] = com_procs[i].get(timeout=timeout)

      return spectra_dom,spectra_com

#Attempt to find the color temp in the EXIF
def find_temp(img):
    try :
        EXIF  = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in img._getexif().items()
        if k in PIL.ExifTags.TAGS
        }
        if 'WhiteBalance' in EXIF.keys():
            temp = EXIF['WhiteBalance']
        else : temp = 0
    except AttributeError : temp = 0
    return temp

#Colorize the histogram roughly like the visible spectra
def colorize_hist(hist):
    N,bins,patches = hist
    bins[bins<vmin] = vmin
    bins[bins>vmax] = vmax
    # We'll color code by height, but you could use any scalar
    fracs = bins / bins.max()
    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = cs.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = cmap(norm(thisfrac))
        thispatch.set_facecolor(color)



'''****************************************************************************
* Main Loop
****************************************************************************'''
if __name__ == '__main__':
  #Open a file dialog to select the image, then load it
  image_path = get_file_path()
  img        = Image.open(image_path)


  #Attempt to find the color temp of the image and move it from the default
  temp = find_temp(img)
  if temp:
          print(f"found photo temp: {temp}K")
          Temperature = temp

  #Resize the image to something more workable
  r,c   = img.size
  img   = img.resize(map(int,[r*scale,c*scale]))
  RGB   = np.asarray(img)


  #Convert to XYZ->xyY colorspace
  XYZ  = colour.sRGB_to_XYZ(RGB / 255)
  xy   = colour.XYZ_to_xy(XYZ)
  xy_n = colour.CCT_to_xy(Temperature)

  #Calculate the dominant wavelength of the color
  spectra_dom,spectra_com = domcom_spectra(xy,xy_n)


  #Plot the CIExyY colorpsace and our pixels on it
  CIExyY = plt.figure("CIExyY colorspace")

  cmfs = colour.CMFS['CIE 1931 2 Degree Standard Observer']
  i,j  = colour.XYZ_to_xy(cmfs.values).T
  RGB  = d.chromaticity_diagram_colours_CIE1931()
  x,y  = xy.T

  scatter = plt.scatter(i,j,marker='o',label='Wavelengths (nm))')
  pixels  = plt.plot(x.flatten(),y.flatten(),'.',color='black',label='Pixels',alpha=0.8)
  cspace  = plt.imshow(RGB,extent=[0,1,0,1],zorder=0)
  #Set the plotting limits to best reflect the space and label
  plt.xlim(0.0,0.75)
  plt.ylim(0.0,0.85)
  plt.xlabel("x chromaticity")
  plt.ylabel("y chromaticity")
  plt.legend()


  #Plot the Color Matching Functions used
  wavelength,LMS  = cmfs.domain,cmfs.range
  colors          = ['red','green','blue']
  labels          = ['X','Y','Z']
  plt.figure("Color Matching Functions Used")
  [plt.plot(wavelength,LMS[:,i],color=colors[i],label=labels[i]) for i in range(LMS.shape[1])]
  plt.xlabel("Wavelength (nm)")
  plt.ylabel("Luminosity")
  plt.legend()


  #Plot the image used for the calcs
  plt.figure("Image used to Process")
  plt.imshow(img)
  plt.axis('off')


  #Plot the Histograms of the Dominant and Complimentary wavelengths
  fig,ax   = plt.subplots(1,2,sharey=True,tight_layout=True)
  hist_dom = ax[0].hist(spectra_dom.flatten(), 500, density=True, label='Dom',alpha=0.7,log=True)
  hist_com = ax[1].hist(spectra_com.flatten(), 500, density=True, label='Com',alpha=0.7,log=True)
  #Attempt to match spectr to color
  colorize_hist(hist_dom)
  colorize_hist(hist_com)
  #Labels
  ax[0].set_title("Dominant Wavelength")
  ax[0].set_xlabel("Wavelength (nm)")
  ax[0].set_ylabel("Spectral Density")
  ax[1].set_title("Complimentary Wavelength")
  ax[1].set_xlabel("Wavelength (nm)")


  #Plot image of dominant wavelength
  plt.figure("Dominant Wavelength Image")
  plt.imshow(spectra_dom,cmap=cmap,vmin=vmin,vmax=vmax)
  plt.axis('off')

  #Plot image of complimentary wavelegth
  plt.figure("Complimentary Wavelength Image")
  plt.imshow(spectra_com,cmap=cmap,vmin=vmin,vmax=vmax)
  plt.axis('off')

  #Display everything
  plt.show()
