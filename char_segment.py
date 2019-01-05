# Importing required libraries
import numpy as np
from keras.models import load_model
from char_segment_utils import get_box_lines, resize_space_widths, get_letters_and_spaces, predict_text

# Loading model weights
model = load_model('model_weights_26.h5')

while True:

	path = input('Enter image path: ')
	box_lines,space_widths_lines,box_height_lines = get_box_lines(path)
	resized_space_widths_lines = resize_space_widths(space_widths_lines, box_height_lines)
	letters_lines, intra_box_spaces_lines = get_letters_and_spaces(box_lines, resized_space_widths_lines)
	predict_text(letters_lines,intra_box_spaces_lines,model)