import cv2
import numpy as np
import math
import open3d
from Chessboard import (threshold_finder,
                        draw_corners)

test = False  # Other tests
test3d = True # Almost real time 3D reconstruction

planeH = 13
planeW = 23
destination = np.array([[[0, 0]], [[planeW, 0]], [[planeW, planeH]], [[0, planeH]]])



def load_coefficients():
    coeffiecients = cv2.FileStorage("intrinsics.xml", cv2.FILE_STORAGE_READ)
    K = coeffiecients.getNode("K").mat()
    dist = coeffiecients.getNode("dist").mat()
    K_inv = np.linalg.inv(K)
    
    return K , K_inv , dist



def goodfeatures(img, mask):
    return cv2.goodFeaturesToTrack(img, maxCorners=4, qualityLevel=0.001, minDistance=50, mask=mask, blockSize=5, gradientSize=7)



def sort_corners(board_corners):
    #Sort corners starting from top left
    center = np.sum(board_corners, axis=0) / len(board_corners)
    sorted_corners = sorted(board_corners, key=lambda p: math.atan2( p[0][0] - center[0][0], p[0][1] - center[0][1]), reverse=True)
    
    return np.roll(sorted_corners, 1, axis=0)



def get_Boards(thresh, ori):
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask_top = np.zeros(thresh.shape, np.uint8)
    mask_buttom = mask_top.copy()
    
    # Need to find the right contours with a search
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:10]

    # I manually searched for the desired contours
    desidered_contours = []
    for i in contours:
        if cv2.contourArea(i) < 130000:
            desidered_contours.append(i)
            
    board_top = desidered_contours[0]
    board_buttom = desidered_contours[1]
    
    cv2.drawContours(mask_buttom, [board_buttom], 0, 255, -1)
    cv2.drawContours(mask_top, [board_top], 0, 255, -1)
        
    # find corners
    corners_buttom = goodfeatures(thresh, mask_buttom)
    corners_top = goodfeatures(thresh, mask_top)
    #and sort them
    corners_buttom = sort_corners(corners_buttom)
    corners_top = sort_corners(corners_top)
    
    mask = mask_buttom + mask_top # can be used for debugging

    ''' START DEBUG '''
    if test:
        frame = ori.copy()
        cv2.drawContours(frame, [board_top.astype(np.int32)], -1, (0, 0, 255),2)
        cv2.drawContours(frame, [board_buttom.astype(np.int32)], -1, (0, 0, 255),2)
        draw_corners(frame, corners_buttom, color = (0, 200, 0), label = "board buttom")
        draw_corners(frame, corners_top, color = (0, 200, 0), label = "board buttom + top")
    ''' END DEBUG '''
    
    return np.array(corners_buttom), np.array(corners_top)



def get_Plane(Homography, k_inv):

    # Apply the inverse intrinsics
    reverse = np.matmul(k_inv, Homography)
    reverse /= np.linalg.norm(reverse[:, 1]) # Normalize by by vector of scale h1
    r1, r2, t = np.hsplit(reverse, 3)
    r3 = np.cross(r1.T, r2.T).T  # exactly the normal

    w, u, vt = cv2.SVDecomp(np.hstack([r1, r2, r3]))
    R = np.matmul(u, vt) # rotation matrix

    # t is the point of the plane. Having also the normal we have exactly the plane
    point = t[:, 0]
    normal = R[:, 2]
    
    return point, normal



def find_all_red_points(frame):
    # To be more insensitive to lighting we can look for red hues after converting to HSV colorspace. 
    # But since red has the same 0 hue as black/gray/white, I inverted the image so that red becomes cyan.
    hsv = cv2.cvtColor(~frame, cv2.COLOR_BGR2HSV)
    # Set low and high limit for the tones we want to identify - based on Hue of cyan=90 since is better tracked than redV
    lower_red = np.array([35, 20, 50])
    upper_red = np.array([110, 255, 255])
    # Mask all red pixels
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # We give the slider a size, let's say 4 x 4 pixels. What happens is we slide this slider around, and if all of the pixels are white, then we get white, otherwise black.
    kernel = np.ones((3, 4), np.uint8)
    # The next pair is "opening". The goal with opening is to remove "false positives" so to speak. Sometimes, in the background, you get some pixels here and there of "noise.
    image = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Store the actual points found
    points = cv2.findNonZero(image)
    
    ''' START DEBUG '''
    if test:
        cv2.imshow('mask', mask)
        cv2.waitKey(1)
    ''' END DEBUG '''
    
    return image, points



def find_red_points_inside(board_desk, board_wall, img_laser):
    # Create new zero values Mat that has the same size of your original image
    mask_desk = np.zeros(img_laser.shape[:2], dtype=np.uint8)
    mask_wall = np.zeros(img_laser.shape[:2], dtype=np.uint8)
    # Draw the boards on it with fillConvexPoly
    cv2.fillConvexPoly(mask_desk, np.array(board_desk, 'int32'), 255)
    cv2.fillConvexPoly(mask_wall, np.array(board_wall, 'int32'), 255)
    # Bitwise_and this image with your original mask and apply findnonzero function on the result image
    points_desk = cv2.findNonZero(cv2.bitwise_and(img_laser, mask_desk))
    points_wall = cv2.findNonZero(cv2.bitwise_and(img_laser, mask_wall))

    return points_desk, points_wall



def set_homogeneous(points):
    #add a column of ones to make a series of points homogeneous
    ones = np.ones(points.shape[0]).reshape(points.shape[0], 1)
    points = np.hstack((points[:, 0], ones))
    
    return points



def linePlaneCollision(plane_point, plane_normal, direction):
    # Line plane intersection implementation d = (p0 -l0) * n/ (l*n) where:
    # l0 is a point on the line for simplicity 0,0,0
    # p0 is a point on the plane
    # n the normal
    # l is a vector direction of the line (exiting ray) 
    # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection'''
    d = np.dot(plane_point, plane_normal) / np.dot(direction, plane_normal)
    point_of_intersection = direction * d
    
    return point_of_intersection



def do_intersection(ray_Directions, plane):
    #Given a plane and a series of rays, get the intersection

    intersection = [linePlaneCollision(plane[0], plane[1], rayDirection) for rayDirection in ray_Directions]
    
    return intersection



def fit_plane(points):
    # Fitting a plane to many points in 3D
    # https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
    
    mean = np.mean(points, axis= 0)
    coord_xx = 0
    coord_xy = 0
    coord_xz = 0
    coord_yy = 0
    coord_yz = 0
    coord_zz = 0
    
    #Compute the linear system matrix and vector elements
    for i in points:
        diff = i - mean
        coord_xx += diff[0] * diff[0]
        coord_xy += diff[0] * diff[1]
        coord_xz += diff[0] * diff[2]
        coord_yy += diff[1] * diff[1]
        coord_yz += diff[1] * diff[2]
        coord_zz += diff[2] * diff[2]
        
    #Solve linear system
    det_x = coord_yy * coord_zz - coord_yz * coord_yz
    det_y = coord_xx * coord_zz - coord_xz * coord_xz
    det_z = coord_xx * coord_yy - coord_xy * coord_xy
    
    det_max = max(det_x, det_y, det_z)
    
    # Compute the fitted lineh ( x ) = barH + barA ∗ ( x − barX ).
    if det_max == det_x:
        normal = np.array([
                           det_x,
                           coord_xz * coord_yz - coord_xy *coord_zz,
                           coord_xy * coord_yz - coord_xz * coord_yy])
    elif det_max == det_y:
        normal = np.array([
                            coord_xz * coord_yz - coord_xy * coord_zz,
                            det_y,
                            coord_xy * coord_xz - coord_yz * coord_xx])
        
    else:
        normal = np.array([
                           coord_xy * coord_yz - coord_xz * coord_yy,
                           coord_xy * coord_xz - coord_yz * coord_xx,
                           det_z]
                         )
    #normalize
    normal = normal / np.linalg.norm(normal)
    centroid = np.array(mean)
    
    return centroid, normal



def get_original_colors(points, reference_frame):
    # Open3d works with RGB but openCV with BGR, so I need to convert colors,
    # also Open3d has colors in range from 0 to 1 and not 255
    points = points.squeeze(1)
    rgb = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2RGB)
    rgb = rgb/255
    rgb = rgb[ points[:, 1], points[:, 0] ].astype(np.float64)
    
    return rgb


    
if __name__ == '__main__':
    
    # Load camera matrix and distortion coefficients
    K, K_inv, dist = load_coefficients()
  
    pcd = open3d.geometry.PointCloud()
    #open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Debug)
    
    ''' START 3D PARAMETERS '''
    if test3d:
        initiator = np.zeros(6).reshape(2, 3)
        pcd.points = open3d.utility.Vector3dVector(initiator)
        pcd.colors = open3d.utility.Vector3dVector(initiator)
        open3d.io.write_point_cloud("test.pcd", pcd)
        source = open3d.io.read_point_cloud("test.pcd")
        vis = open3d.visualization.Visualizer()
        vis.create_window()      
        vis.add_geometry(source)
        threshold = 0.05
        icp_iteration = 1
        save_image = False
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    ''' END 3D DEBUG SETUP '''
    
    # Only for the first frame Find parameters of desk plane and wall plane we need to:
        # Detect the rectangles corners
        # Find their homography
        # Split them into rotation and translation.
        # Find plane desk and plane wall
        
    #cap = cv2.VideoCapture("./videos/puppet.mp4") 
    #cap = cv2.VideoCapture("./videos/soap.mp4") 
    #cap = cv2.VideoCapture("./videos/cup1.mp4") 
    cap = cv2.VideoCapture("./videos/cup2.mp4") 
    

    _, frame = cap.read()
    reference_frame = cv2.undistort(frame, K, dist)
    gray_frame = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    
    import time
    start_time = time.time()
    thresh_frame = threshold_finder(gray_frame)
    # Recover Homographies of the two planes--> to do so I need first the rectangles and cordinates of both the planes
    board_desk, board_wall = get_Boards(thresh_frame,reference_frame)

    # Now I can find Homography of the desk
    destination = sort_corners(destination)
    homography_desk = cv2.findHomography(destination, np.array(board_desk))[0]
    homography_wall = cv2.findHomography(destination, np.array(board_wall))[0]
    
    # Get the plane througout rotation and translation vectors
    # [0] is the point, [1] is the normal
    plane_desk = get_Plane(homography_desk, K_inv) #R3
    plane_wall = get_Plane(homography_wall, K_inv)
    
    # Storage for 3D points and object colors
    points_rgb = []
    points_obj = []
    counter = 0
    
    #Start main cicle

    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            if test3d:    
                vis.destroy_window()
            cv2.destroyAllWindows()
            break
        ''' START 3D DEBUG '''
        if test3d and counter % 60 == 0:
            source = open3d.io.read_point_cloud("test.pcd")
            source.transform(flip_transform)
            for i in range(icp_iteration):
                vis.add_geometry(source)
                vis.poll_events()
                vis.update_renderer() 
        ''' END 3D DEBUG '''   
                
        # Undistort frame
        current_frame_undistorted = cv2.undistort(frame, K, dist)
        # Find all red points inside image
        laser_thresh, laser_points = find_all_red_points(current_frame_undistorted)
        # Find red points inside both boards
        imagePointsDesk, imagePointsWall = find_red_points_inside(board_desk, board_wall, laser_thresh)
        
        
        if imagePointsDesk is not None and imagePointsWall is not None:

            homogeneous_coord_desk = set_homogeneous(imagePointsDesk)
            homogeneous_coord_wall = set_homogeneous(imagePointsWall)

            # Exit rays ray_wall and ray_desk in space obtained by k_inv * homogeneous coord to do so we do:
            # For each red point, create corresponding exiting ray: 
            ray_wall = [np.matmul(K_inv, i) for i in homogeneous_coord_wall]
            ray_desk = [np.matmul(K_inv, i) for i in homogeneous_coord_desk]
            
            
            # Each ray exiting from the camera center, passing through the imaged red pixels in the rectangles in the image plane,
            # intersects the wall and desk planes (represented with point+normal) in the red pixels. This means that we can obtain the 3D positions of the
            # red pixels in the real world (desk plane and wall plane) by intersecting the exiting rays (from the camera center, passing through the imaged red pixels) with the wall and desk planes.
            #Now we need to intersect the 3D line in space with the 3D laser plane in space.
            
            intersect_wall = do_intersection(ray_wall, plane_wall)
            intersect_desk = do_intersection(ray_desk, plane_desk)
            

            if intersect_wall is not None and intersect_desk is not None:

                # Now this point we need to find the laser plane
                # We can then fit a plane between the two lines, because the points are coplanar by definition (they come from a single laser that projects a line).
                # This plane is exactly the laser plane in 3D 
     
                intersect_desk_wall = intersect_desk + intersect_wall
                # For fitting a plane between the two lines see Least Squares Fitting of Data by Linear or Quadratic Structures
                laser_plane = fit_plane(intersect_desk_wall)
                                
                laser_points_h = set_homogeneous(laser_points)
                
                #Exit rays ray_plane
                ray_tot = [np.matmul(K_inv, i) for i in laser_points_h]
                
                # The intersection between each ray and the laser plane is a 3D point composing the scene that we want to reconstruct.
                # In other words, once we have the laser plane expressed with a (point, normal) tuple, we can reconstruct the points by intersecting rays exiting from the camera with the laser plane    
                laserPoints3D = do_intersection(ray_tot, laser_plane)
                
                #Get RGB colors from the original image
                colors = get_original_colors(laser_points, reference_frame)
                
                points_rgb.extend(colors)
                points_obj.extend(laserPoints3D)
                
                ''' START 3D DEBUG '''
                if test3d and counter % 60 == 0:    
                    pcd.points = open3d.utility.Vector3dVector(points_obj)
                    pcd.colors = open3d.utility.Vector3dVector(points_rgb)
                    open3d.io.write_point_cloud("test.pcd", pcd)
                counter += 1
                ''' END 3D DEBUG '''
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("--- %s seconds ---" % (time.time() - start_time))      
    pcd.points = open3d.utility.Vector3dVector(points_obj)
    pcd.colors = open3d.utility.Vector3dVector(points_rgb)
    print("\n Saving point cloud to result.ply")
    open3d.io.write_point_cloud("result.ply", pcd)
    print("\n Visualizing geometry")
    open3d.visualization.draw_geometries([pcd])
    