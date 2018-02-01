import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    #standard_im = np.copy(image)
    standard_im = cv2.resize(image, (32, 32))
    row_crop = 6
    col_crop = 10
    standard_im = standard_im[row_crop:-row_crop, col_crop:-col_crop, :]
    return standard_im

def hsv_conversion(rgb_image):
	return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

def one_hot_encode(label):
    
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = [0, 0, 0]
    if label == "red":
        one_hot_encoded[0] = 1
    elif label == "yellow":
        one_hot_encoded[1] = 1 
    elif label == "green":
        one_hot_encoded[2] = 1
    #else:
        #raise TypeError('Please input red, yellow, or green. \
        #    Your input is:', label)

    return one_hot_encoded

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

def hsv_histograms(rgb_image):
    # Convert to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Create color channel histograms
    h_hist = np.histogram(hsv[:,:,0], bins=32, range=(0, 180))
    s_hist = np.histogram(hsv[:,:,1], bins=32, range=(0, 256))
    v_hist = np.histogram(hsv[:,:,2], bins=32, range=(0, 256))
    
    # Generating bin centers
    bin_edges = h_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

    # Plot a figure with all three histograms
    fig = plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.bar(bin_centers, h_hist[0])
    plt.xlim(0, 180)
    plt.title('H Histogram')
    plt.subplot(132)
    plt.bar(bin_centers, s_hist[0])
    plt.xlim(0, 256)
    plt.title('S Histogram')
    plt.subplot(133)
    plt.bar(bin_centers, v_hist[0])
    plt.xlim(0, 256)
    plt.title('V Histogram')
    plt.show()
    
    return h_hist, s_hist, v_hist

def brightness_feature(rgb_image):
    # Convert to HSV colorspace
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    # Apply mask to get brighter pixels (where the light is)
    lower_mask = np.array([0,25,125])
    upper_mask = np.array([180,235,235])
    mask = cv2.inRange(hsv, lower_mask, upper_mask)
    res = cv2.bitwise_and(hsv,hsv, mask= mask)
    # We are returning and HSV image with the applied mask
    feature = res
    
    return feature

def masked_hue(hsv_image):
	# define masks ranges 
	# R 160-180
	lower_red = np.array([160])
	upper_red = np.array([180])
	r_mask = cv2.inRange(hsv_image[:,:,0], lower_red, upper_red)
	# Y 10-30
	lower_yellow = np.array([10])
	upper_yellow = np.array([30])
	y_mask = cv2.inRange(hsv_image[:,:,0], lower_yellow, upper_yellow)
	# G 80-100
	lower_green = np.array([80]) 
	upper_green = np.array([100])
	g_mask = cv2.inRange(hsv_image[:,:,0], lower_green, upper_green)
	# Combine masks
	mask = r_mask + y_mask + g_mask
	# Copy H channel
	masked_image = np.copy(hsv_image[:,:,0])
	masked_image[mask == 0] = [0]
	#print("Masked:", masked_image)
	return masked_image

def mode_hue(hsv_image):
    import scipy.stats
    masked_im = masked_hue(hsv_image)
    total_mode = scipy.stats.mode(masked_im[np.nonzero(masked_im)])[0]
    #print('Total MODE: ', total_mode)
    return total_mode[0] if len(total_mode)>0 else 0

def median_hue(hsv_image):
    masked_im = masked_hue(hsv_image)
    nonzero = masked_im[np.nonzero(masked_im)]
    if len(nonzero)>0:
        return np.median(masked_im[np.nonzero(masked_im)])
    return 0

def estimate_label(rgb_image):
    masked_image = brightness_feature(rgb_image)
    mode = mode_hue(masked_image)
    median = median_hue(masked_image)
    print('Mode: ', mode, ', Median: ', median)
    hue = median
    predicted_label = one_hot_encode("undefined")
    if hue >= 160. and hue <= 180.:
        predicted_label = one_hot_encode("red")
    elif hue >= 10. and hue <= 30.:
        predicted_label = one_hot_encode("yellow")
    elif hue >= 80. and hue <= 100.:
        predicted_label = one_hot_encode("green")
    return predicted_label

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    i = 0
    for image in test_images[0:200]:
        i = i +1 
        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        print('----------------------')
        predicted_label = estimate_label(im)
        print(i, true_label, predicted_label)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels

#### Main ####

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

### 1. Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('--- misclassifieds: ')
for image in MISCLASSIFIED:
    print('---')
    masked_image = brightness_feature(image[0])
    print('Masked image:')
    print(masked_image[:,:,0])
    print('Mode:', mode_hue(masked_image))
    print('Median:', median_hue(masked_image))
    print('Pred: ', image[1], ', True: ', image[2])
    plt.imshow(image[0])
    plt.show()

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

# import test_functions
# tests = test_functions.Tests()

# if(len(MISCLASSIFIED) > 0):
#     # Test code for one_hot_encode function
#     tests.test_red_as_green(MISCLASSIFIED)
# else:
#     print("MISCLASSIFIED may not have been populated with images.")

#####
number_of_misclassified = len(MISCLASSIFIED)
i = 0

for item in range(number_of_misclassified ):
    selected_image = MISCLASSIFIED[item]
    predicted_label, true_label = selected_image[1], selected_image[2]    

    if true_label == one_hot_encode("red") and predicted_label == one_hot_encode("green"):
        plt.imshow(selected_image[0])
        print("where in misclassified list:", item)
        i+=1

print("Number of red lights mistaken green:", i, " of ", number_of_misclassified, 
       "misclassified images")