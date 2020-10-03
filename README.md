# 3D-Laser-Scanner

### Calibrating the Camera
python Chessboard.py​ will produce the intrinsics.xml file and calibrate the camera.
To debug the program the file contains the following variables:
- **test boolean variable**
- **modulus** - shows only an amount of images

### 3D laser scanner
python 3Dscanner.py ​will execute the laser scanner.
To debug the program the file contains the following variables:
-**test boolean variable**
-**test3d boolean variable which executes an almost real time processing**

File names can be changed in the main of the 3Dscanner.py file.
Dependencies
The following packages have been used:
- **Opencv**
- **Numpy**
- **Math**
- **Open3d**
- **glob3**

### Processing times
The test were performed on Windows 10 build 19041: i7 8700k, 24b Ram DDR4
According to the data obtained below, and knowing that the videos are 15FPS, we can say
that on average we have a processing time per frame of 0.082s.

| File name  | Video length | Total Processing Time | Frames Processing | Debug Proc. Time  |
| ---------- | ------------ | --------------------- | ----------------  | ----------------  |
| cup1.mp4   | 62s          | 87s                   | 78s               | 125s              |
| cup2.mp4   | 45s          | 66s                   | 58s               | 88s               |
| puppet.mp4 | 53s          | 73s                   | 64s               | 121s              |
| soap.mp4   | 72s          | 97s                   | 87s               | 144s              |

Activating the real time processing slows down the procedure by a significant bit. However
this was expected, according to the documentation found on open3d.

