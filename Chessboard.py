import cv2
import math
import numpy as np
import glob


dirpath = "calibration/calib/"
prefix = "img_000"
image_format = "png"
img_path = glob.glob(dirpath+'/' + prefix + '*.' + image_format)

#debug variables
test = True
counter = 0
modulus = 30

warpedW = 700
warpedH = 900
destination = np.array([[[0, 900]], [[0, 0]], [[700, 0]],[[700, 900]]])
corners = ( [[1, j,0] for j in range(1, 12, 2)]
                       + [[i, j,0] for i in range(3, 14, 2) for j in range(1, 14, 2)]
                       + [[15, j,0] for j in range(1, 12, 2)]
                       + [[17, j,0] for j in range(3, 10, 2)]
                       )
corners = np.expand_dims(np.array(corners), axis=1).astype(np.float32)



def draw_corners(image, points, color = (150, 150, 150), label = "" ):
    count = 0
    for c in points:
        px, py = c[0][0], c[0][1]
        cv2.circle(image, (px, py), 5, color , -1)
        cv2.putText(image,text = str(count), org = (px,py),fontFace = 1, fontScale = 1.2, color = color)
        count += 1
    cv2.imshow("Sorted points " + label, image)
    cv2.waitKey(0)



def threshold_finder(images):
	threshold = cv2.threshold(images,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	return threshold



def sort_corners(board_corners):
    #Sort corners starting from top left
    center = np.sum(board_corners, axis=0) / len(board_corners)
    sorted_corners = sorted(board_corners, key=lambda p: math.atan2( p[0][0] - center[0][0], p[0][1] - center[0][1]), reverse=True)
    return sorted_corners



def sort_corners_xy( mask, coord_xy):

        #Given the mask, finds the four corners of the rectangle
        chessboard_corners = cv2.goodFeaturesToTrack( mask, maxCorners=4, qualityLevel=0.01, minDistance=50, mask = None, blockSize=5, gradientSize = 3)
        
        dist = np.Inf
        closest = np.Inf
        
        #Given the location of the x_y corner, finds the chessboard corner closest to it
        for distance in chessboard_corners:
            temp = cv2.norm(coord_xy,  distance, cv2.NORM_L2)
            if temp < dist:
                dist = temp
                closest = distance
        
        #Sort corners in clockwise order
        sorted_corners = sort_corners(chessboard_corners)        

        #re-arange the corners until the closest to x_y is the first one 
        while any(sorted_corners[0].ravel() != closest.ravel()):
            sorted_corners = np.roll(sorted_corners, 1)

        return sorted_corners



def get_board_corners(img_gray):
    img_thresh = threshold_finder(img_gray)
    #Create all the possible contours
    cnts, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Sort contours based on their area, and keep only the first 50 ones
    areas = sorted(cnts, key = cv2.contourArea, reverse = True)[:50]
    # I manually selected the third contour since is the one corresponding to the inner rectangle chessboard
    chessboard = areas[2]
    #Create a mask with the selected contour
    mask = np.zeros(img_thresh.shape, np.uint8)
    cv2.drawContours(mask, [chessboard], 0, 255, -1)
    
    coord_xy = None
    for cnt in cnts:
        if (35 < cv2.contourArea(cnt) < 350): #check the manually selected possible area for x_y figure
            if cv2.pointPolygonTest(chessboard, tuple(cnt[0][0]), True) > 1: #Check the distance between contours and the point found
                #test = cv2.contourArea(cnt)
                #test = cv2.pointPolygonTest(chessboard, tuple(cnt[0][0]), True)
                
                '''START DEBUG'''
                if test and counter % modulus == 0:
                    img_grayC = img_gray.copy()
                    cv2.drawContours(img_grayC, [cnt], -1, (0, 100, 0), 2)
                    cv2.imshow("XY Contour",img_grayC)
                    cv2.waitKey(0)
                '''END DEBUG'''
                
                coord_xy = cnt[0].astype(np.float32)
                break #Consider the first found as a good one.
    #Sort board corners starting from the xy reference coordinate
    board_corners = sort_corners_xy(mask, coord_xy)
    
    return board_corners, chessboard



def sort_chessboard(chess_corners):
        
        sorted_corners = []
        
        def takefirst(array):
            return array[0]
        
        for i in np.arange(start = warpedH, stop = 0, step = -100):
            #Takes all the corners found in a certain Height, starting from the buttom
            x = chess_corners[np.logical_and(chess_corners[:, :, 1] < i, chess_corners[:, :, 1] > i - 100)]
            #Sort them from left to right
            x = sorted(x, key=takefirst)
            sorted_corners += x
        sorted_corners = np.array(sorted_corners)    
        return np.expand_dims(sorted_corners, axis=1).astype(np.float32)



def get_chessboard_corners(chessboard):
    
    mask = np.full(chessboard.shape,255, np.uint8)
    # I hide x_y from the image to avoid to be classified as a possible chessboard corner
    cv2.rectangle(mask, (0, warpedH-50), (30, warpedH), 0, -1)
    cv2.rectangle(mask, (0, warpedH - 30), (warpedW - 640, warpedH), 0, -1,)

    '''if test and counter % modulus == 0:
        cv2.imshow('',mask)
        cv2.waitKey(0)'''
    
    #Find corners
    chess_corners = cv2.goodFeaturesToTrack(chessboard ,maxCorners=58, qualityLevel=0.001, minDistance=70, mask=mask, blockSize=3, gradientSize=13)
    #Refine corners with cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 20, 0.001)
    chess_corners = cv2.cornerSubPix(chessboard, chess_corners, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
    
    sorted_corners = sort_chessboard(chess_corners)
    
    ''' START DEBUG '''
    if test and counter % modulus == 0:
        opening_copy = chessboard.copy()
        draw_corners(opening_copy,sorted_corners, label = "chessboard")
    ''' END DEBUG '''
   
    return sorted_corners


    
def find_homography(board_corners, destination):
    H = cv2.findHomography(np.array(board_corners), destination)[0]
    H_inv = np.linalg.inv(H)
    return H, H_inv



def find_warp(board_corners, destination):
        H , H_inv = find_homography(board_corners, destination)
        warp = cv2.warpPerspective(img, H, (warpedW, warpedH))
        warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        warp = threshold_finder(warp)
        return warp



if __name__ == '__main__':
    
    imagePoints = []
    objectPoints = []
    
    for i  in img_path:
        img = cv2.imread(i)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Find the board corners and the countour
        board_corners, chessboard = get_board_corners(img_gray)
        
        #Warp the image given the board_corners and the original position
        warp = find_warp(board_corners, destination)        
        
        #Find the chessboard corners
        chessboard_corners = get_chessboard_corners(warp)
        
        #Find the homography_inv to perform the prespective transformation
        _, H_inv = find_homography(board_corners,destination) 
        chessboard_inner = cv2.perspectiveTransform(chessboard_corners, H_inv)
        
        ''' START DEBUG '''
        if test and counter % modulus == 0:
            image_copy = img.copy()
            cv2.drawContours(image_copy, [chessboard], -1, (0, 100, 0), 2)
            draw_corners(image_copy, board_corners, color = (0, 200, 0), label = "rectangle")
            draw_corners(image_copy, chessboard_inner, color = (0, 100, 0), label = "original image N." + str(counter))
        counter+=1
        ''' END DEBUG '''
        
        if(len(chessboard_inner)==58):
            imagePoints.append(chessboard_inner)
            objectPoints.append(corners)
            
    imagePoints = np.array(imagePoints, dtype=np.float32)
    objectPoints = np.array(objectPoints, dtype=np.float32)
    
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, (img.shape[0], img.shape[1]), None, None)
    print("\n RMS : \n")
    print(ret)
    print("\n K : \n")
    print(K)
    print("\n dist : \n")
    print(dist)
    print("\n rvecs : \n")
    print(rvecs)
    print("\n tvecs : \n")
    print(tvecs)
    print("\n Images used: " + str(imagePoints.shape[0]) + "/50 \n")
    
    # Save intrinsics that we will need
    Kfile = cv2.FileStorage("intrinsics.xml", cv2.FILE_STORAGE_WRITE)
    Kfile.write("RMS", ret)
    Kfile.write("K", K)
    Kfile.write("dist", dist)
    Kfile.release()
    print("The file intrinsics.xml has been saved")