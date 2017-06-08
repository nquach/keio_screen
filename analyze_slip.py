import numpy as np
import os
import tifffile as tiff
from skimage.io import imread
from skimage.measure import label, regionprops
import scipy
import matplotlib.pyplot as plt
from analyze import analyze_slip_pos
from gaussian_model import extract_infected
from sklearn import mixture
from keio_names import get_keio_names, loc_to_strain, pos_to_strain
import openpyxl as xls

#directory with infection data
infect_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/5.15.17_SLIP/infection/'
#directory with control data
control_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/5.15.17_SLIP/no_infection/'
#2
mask_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/5.15.17_SLIP/masks/'
control_mask_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/5.15.17_SLIP/control_masks/'
save_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/deep_learning/plots/6.8.17/'
excel_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/5.15.17_SLIP/'
#pos = 'A7'
plate_num = 9

plate = (plate_num-1)/2

gaussian_confidence = 0.99999

ratio_matrix = analyze_slip_plate(infect_direc, control_direc, save_direc, mask_direc, pos, num_pos = 25, confidence = 0.75, gaussian_confidence = gaussian_confidence, multiplier = 1.5, verbose = True, plot = True, save_direc = save_direc)

#Reverse the column numbers!
ratio_matrix = np.flip(ratio_matrix, 1)

ratio_list = list(ratio_matrix.flatten())

strain_matrix = get_keio_names()

strain_list = list(strain_matrix[plate,:,:].flatten())

print 'Lysis ratios calculated. Saving to excel spreadsheet.'
wb = xls.Workbook()
sheet = xls.get_sheet_by_name('Sheet')

counter = 0
for row in range(1, len(ratio_list)+1):
	cell1 = 'A' + str(row)
	sheet[cell1] = strain_list[counter]
	cell2 = 'B' + str(row)
	sheet[cell2] = ratio_list[counter]

excel_path = os.path.join(excel_direc, 'strain_ratio.xlsx')
wb.save(excel_path)


