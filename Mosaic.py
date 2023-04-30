# Import all modules needed for this script
import sys
import os
import cv2
import numpy as np
import random
from math import floor, ceil


'''
Arguments : -> source - source image, which is the image to be warped
			-> destination - destination image, which forms the reference frame for warping 
			
Returns : -> source_points - points matched in source image
		  -> destination_points - points matched in destination image
		  -> plot - plot of the matchings
		  
Description : This function finds SIFT features between two images and creates an image point map using them
'''
def SIFTmapping(source, destination):
	# Matching threshold used in ratio test
    threshold=0.5
    
    # Find sift features in both images
    sift = cv2.xfeatures2d.SIFT_create()
    keypoint_destination, descriptor_destination = sift.detectAndCompute(destination, None)
    keypoint_source, descriptor_source = sift.detectAndCompute(source, None)

	# Perform brute force matching with 2-NN
    matches = cv2.BFMatcher().knnMatch(descriptor_source,descriptor_destination,k=2)

    # Apply ratio test using the threshold to filter matches
    good_matches = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good_matches.append(m)

    # Create a map between source and destination images if only number of good matches are atleast 4
    if len(good_matches) >= 4:
       source_points = np.float32([ keypoint_source[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
       destination_points = np.float32([ keypoint_destination[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

	# If the number of good matches are more than 50, sample 50 of them randomly and plot the map
    if len(good_matches) >= 50:
        matches_50 = random.sample(good_matches, 50)
        plot = cv2.drawMatches(source,keypoint_source,destination,keypoint_destination,matches_50,None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        plot = cv2.drawMatches(source,keypoint_source,destination,keypoint_destination,good_matches,None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
    # Return matched points in source image, destination image, mapped visualization
    return source_points, destination_points, plot


'''
Arguments : -> source_points - points matched in source image
		    -> destination_points - points matched in destination image
		    
Returns : -> H - Homography matrix associated with source_points and destination_points

Description : This function is used to find the homography matrix from scratch using points matched in source image and points matched in destination image.
'''
def MyHomography(source_points, destination_points):
    A = []
    b = []
    for i in range(len(source_points)):
    	# Remove axes
        x, y = source_points[i].squeeze()
        u, v = destination_points[i].squeeze()
        
        # Build matrices A and b
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        b.append(u)
        b.append(v)
    
    # Convert them to numpy arrays
    A = np.array(A)
    b = np.array(b)
    
    # Find homography matrix using least squares solution to linear matrix equation
    H = np.linalg.lstsq(A, b, rcond=None)[0]
    H = np.append(H, 1).reshape(3, 3)
    
    #return homography matrix
    return H


'''
Arguments : -> source - source image, which is the image to be warped
			-> destination - destination image, which forms the reference frame for warping 
			-> destination_location - the location in the warped frame to hold the destination image
			
Returns : -> warp_frame - image frame containing the destination image at the specified location

Description : This function is used to create a frame that is large enough to hold the images after stitching them
'''
def CreateWarpFrame(source, destination, offset_main=0, offset_auxilary=0, destination_location='right'):
	# Get the height and width of the source image and destination image
    height_source, width_source = source.shape[:2]
    height_destination, width_destination = destination.shape[:2]
    
    # Create a warp frame using the size of main axis and auxillary axis and place the destination image at the specified position by offseting the image into warp frame
    
    # For destination image to be in right side of warp frame
    if destination_location=='right':
        warp_frame = np.zeros((max(height_destination, height_source)+offset_auxilary, width_destination+offset_main, 3), dtype='uint8')
        warp_frame[:height_destination, offset_main:width_destination+offset_main, :] = destination[:][:][:]
    
    # For destination image to be in left side of warp frame
    elif destination_location=='left':
        warp_frame = np.zeros((max(height_destination, height_source)+offset_auxilary, width_destination+offset_main, 3), dtype='uint8')
        warp_frame[:height_destination, :width_destination, :] = destination[:][:][:]
        
    # For destination image to be in bottom side of warp frame
    elif destination_location=='bottom':
        warp_frame = np.zeros((height_destination+offset_main, max(width_destination, width_source)+offset_auxilary, 3), dtype='uint8')
        warp_frame[offset_main:height_destination+offset_main, :width_destination, :] = destination[:][:][:]
    
    # For destination image to be in top side of warp frame
    elif destination_location=='top':
        warp_frame = np.zeros((height_destination+offset_main, max(width_destination, width_source)+offset_auxilary, 3), dtype='uint8')
        warp_frame[:height_destination, :width_destination, :] = destination[:][:][:]

	# Return the warp frame with destination image at specified location
    return warp_frame


'''
Arguments : -> warp_frame - image frame with destination image at a specified position
			-> source - source image that needs to be warped into the frame 
			-> homography_mat - homography matrix obtained from featured points matched in source and destination image
			-> shift - offset on an axis to align both the images
			-> destination_location - location of the destination image in the warp frame
			
Returns : -> warped_image - image containing the outcome of warping source image into warp frame

Description : This function is used for warping the source image into the warped frame using the homography matrix
'''
def warpImages(warp_frame, source, homography_mat, shift, destination_location):
	# Create a copy of warp frame for manipulation
    warped_image = warp_frame.copy()
    
    # Access the source image pixel by pixel
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
        
        	# Calculate projected coordinates by solving [x'/k y'/k k]^T = H [x y 1]^T
            p = np.array([x,y,1])
            p_prime = np.dot(homography_mat, p)
            x_prime, y_prime = (p_prime[0]/p_prime[2]), (p_prime[1]/p_prime[2])
			
			# Place the projected points according to location of destination image in warp frame
            x_prime += shift * (destination_location == 'right')
            y_prime += shift * (destination_location == 'bottom')
        
        	# Hash table for horizontal axis
            xPos_test = {
                'right': 0 < ceil(x_prime) and ceil(x_prime)+shift < warped_image.shape[1],
                'left': 0 < ceil(x_prime) < warped_image.shape[1]
            }
			
			# Hash table for vertical axis
            yPos_test = {
                'bottom': 0 < ceil(y_prime) and ceil(y_prime)+shift < warped_image.shape[0],
                'top': 0 < ceil(y_prime) < warped_image.shape[0]
            }
            
            # Warp the image based on hash table values
            if yPos_test.get(destination_location, yPos_test['top']) and xPos_test.get(destination_location, xPos_test['left']):
                warped_image[int(y_prime), int(x_prime), :] = source[y, x, :]

	# Returned warped image
    return warped_image


'''
Arguments : -> warped_image - warped image before using inverse warping
			-> source - source image that needs to be warped into the frame 
			-> homography_matInv - inverse of homography matrix obtained from featured points matched in source and destination image
			-> shift - offset on an axis to align both the images
			-> destination_location - location of the destination image in the warp frame
			
Returns : -> filled_image - image containing the outcome inverse warping on warped images

Description : This function is used to remove black pixels from warped images using inverse warping
'''
def invWarpInterpolation(warped_image, source, homography_matInv, shift, destination_location='right'):
	# Create a copy of warped image for manipulation
    filled_image = warped_image.copy()
    
    # Access the warped image pixel by pixel
    for y_prime in range(filled_image.shape[0]):
        for x_prime in range(filled_image.shape[1]):
        	# Check if pixel is black
            if np.any(filled_image[y_prime, x_prime, :]) < 50:
                # Calculate projected coordinates by solving [x'/k y'/k k]^T = H [x y 1]^T for inverse warping
                p_prime = np.array([x_prime-shift*(destination_location=='right'), y_prime-shift*(destination_location=='bottom'), 1])
                p = np.dot(homography_matInv, p_prime)
                x, y = (p[0]/p[2]), (p[1]/p[2])
				
				# Find changes
                x_floor, x_ceil, y_floor, y_ceil = floor(x), ceil(x), floor(y), ceil(y)
                y_ratio = y - y_floor
                x_ratio = x - x_floor
				
				# Hash table containing locations of neighbouring pixels (4 neighbours)
                neighbours = {
                    'bottom_left': [y_floor, x_floor],
                    'top_left': [y_ceil, x_floor],
                    'bottom_right': [y_floor, x_ceil],
                    'top_right': [y_ceil, x_ceil]
                }
				
				# Perform bilinear interpolation using the 4 neighbours
                if 0 < y_ceil < source.shape[0] and 0 < x_ceil < source.shape[1]:
                    function = np.zeros((3), dtype='uint8')
                    for i in range(3):
                        A = np.array([1-x_ratio, x_ratio])
                        B = np.array([[source[neighbours['bottom_left'][0], neighbours['bottom_left'][1], i],
                                        source[neighbours['bottom_right'][0], neighbours['bottom_right'][1], i]], 
                                        [source[neighbours['top_left'][0], neighbours['top_left'][1], i],  
                                        source[neighbours['top_right'][0], neighbours['top_right'][1], i]]])
                        C = np.array([1-y_ratio, y_ratio])

                        function[i] = np.dot(np.dot(A, B), C.T)

                    filled_image[y_prime, x_prime, :] = function
	
	# Returned warped image after inverse warping with no black pixels
    return filled_image


'''
Arguments : None

Returns : None

Description : This function is the start of execution for the script. It acceses all functions defined above to seamlessly stitch two images.
'''
def main():
	# Read the destination image specified the folder (second argument) and file name (fourth argument)
    destination = cv2.imread(os.path.join(sys.argv[1],sys.argv[3]))
    
    # Read the source image specified the folder (second argument) and file name (third argument)
    source = cv2.imread(os.path.join(sys.argv[1],sys.argv[2]))
    
    # Find SIFT features in both images
    source_points, destination_points, plot = SIFTmapping(source=source, destination=destination)
    
    # Find the maximum values in the points found using SIFT
    [max_xSrc, max_ySrc], [max_xDst, max_yDst] = [int(i) for i in np.max(source_points.squeeze(), axis=0)], [int(i) for i in np.max(destination_points.squeeze(), axis=0)]
    
    # Save the mapping image for SIFT features
    cv2.imwrite(os.path.join(sys.argv[1],'Matches.jpg'),plot)
    
    # Find homography matrix using points found
    M = MyHomography(source_points, destination_points)
    
    # Find inverse of homography matrix
    M_inv = np.linalg.inv(M)
    M_inv = (1 / M_inv.item(8)) * M_inv
    
    # Display the matrices found
    print('\nHomography Matrix using implementation')
    print(M)
    print('\nInverse Homography Matrix')
    print(M_inv)
    
    # Read the location of destination image in warped frame (fifth argument)
    destination_location = sys.argv[4]
    
    # Create a warp frame
    warp_frame = CreateWarpFrame(source=source, destination=destination, offset_main=max_xDst, destination_location=destination_location)
    
    # Warp source image onto warp frame
    warped_image = warpImages(warp_frame=warp_frame, source=source, homography_mat=M, shift=max_xDst, destination_location=destination_location)
    
    # Remove black pixels using inverse warping
    stitched_image = invWarpInterpolation(warped_image=warped_image, source=source, homography_matInv=M_inv, shift=max_xDst, destination_location=destination_location)
    
    # Save the warp frame, warped image with black pixels and warped image without black pixels
    cv2.imwrite(os.path.join(sys.argv[1],'OutputFrame.jpg'),warp_frame)
    cv2.imwrite(os.path.join(sys.argv[1],'Stitched.jpg'),warped_image)
    cv2.imwrite(os.path.join(sys.argv[1],'StitchedWithInversion.jpg'),stitched_image)

if __name__ == "__main__":
    main()
    
    

