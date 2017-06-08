import h5py
import tifffile as tiff
from keras.backend.common import _UID_PREFIXES

from cnn_functions import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes, segment_nuclei, segment_cytoplasm, dice_jaccard_indices, run_model
from model_zoo import sparse_bn_feature_net_31x31 as fn 

import os
import numpy as np

def segment_SLIP(data_direc, control_direc, mask_direc, control_mask_direc):
	alphabet = ['A','B','C','D','E','F','G','H']
	columns = range(1,13)
	for row in range(0,8):
		for column in columns:
			pos = alphabet[row] + str(column)
			print "Segmenting Position " + pos
			current_data_direc = os.path.join(data_direc, pos)
			current_control_direc = os.path.join(control_direc, pos)
			current_mask_direc = os.path.join(mask_direc, pos)
			current_control_mask_direc = os.path.join(control_mask_direc, pos)
			#make position directory for masks if necessary
			try:
				os.stat(current_mask_direc)
			except:
				os.mkdir(current_mask_direc)
			try: 
				os.stat(current_control_mask_direc)
			except:
				os.mkdir(current_control_mask_direc)

			segment_SLIP_pos(current_data_direc, current_mask_direc)
			segment_SLIP_pos(current_control_direc, current_control_mask_direc)
			
def segment_SLIP_pos(data_direc, mask_direc):
	data_location = data_direc
	phase_location = mask_direc

	phase_channel_names = ['Phase']#['channel000']

	trained_network_phase_directory = "/home/nquach/DeepCell2/trained_networks/ecoli/ecoli_all/"   

	phase_prefix = "2016-07-20_ecoli_all_31x31_bn_feature_net_31x31_"
	#"2017-02-12_ecoli_90x_31x31_ecoli_90x_feature_net_31x31_"

	win_phase = 30

	image_size_x, image_size_y = get_image_sizes(data_location, phase_channel_names)
	image_size_x /= 2
	image_size_y /= 2

	list_of_phase_weights = []
	for j in xrange(5):
		phase_weights = os.path.join(trained_network_phase_directory,  phase_prefix + str(j) + ".h5")
		print(phase_weights)
		list_of_phase_weights += [phase_weights]

	phase_predictions = run_models_on_directory(data_location, phase_channel_names, phase_location, model_fn = fn, 
		list_of_weights = list_of_phase_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
		win_x = win_phase, win_y = win_phase, std = False, split = False)

	#phase_masks = segment_nuclei(phase_predictions, mask_location = mask_location, threshold = 0.75, area_threshold = 100, solidity_threshold = 0.75, eccentricity_threshold = 0.95)