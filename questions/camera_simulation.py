import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.widgets import Slider, Button

# Initial random matrices
initial_intrinsic_matrix_to_replace = np.random.rand(3,3)
initial_extrinsic_matrix_to_replace = np.random.rand(3,4)
initial_camera_matrix_to_replace = np.random.rand(3,4)

# Setting up the point cloud
file_data_path= "./images/bunny.xyz"
point_cloud = np.loadtxt(file_data_path, skiprows=0, max_rows=1000000)
# center it
point_cloud -= np.mean(point_cloud,axis=0)
# homogenize
point_cloud = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=1)
# move it in front of the camera
point_cloud += np.array([0,0,-0.15,0])

def calculate_camera_matrix(tx, ty, tz, alpha, beta, gamma, fx, fy, skew, u, v):
    """
    This function should calculate the camera matrix using the given
    intrinsic and extrinsic camera parameters.
    We recommend starting with calculating the intrinsic matrix (refer to lecture 8).
    Then calculate the rotational 3x3 matrix by calculating each axis separately and
    multiply them together.
    Finally multiply the intrinsic and extrinsic matrices to obtain the camera matrix.


    :params tx, ty, tz: Camera translation from origin
    :param alpha, beta, gamma: rotation about the x, y, and z axes respectively
    :param fx, fy: focal length of camera
    :param skew: camera's skew
    :param u, v: image center coordinates
    :return: [3 x 4] NumPy array of the camera matrix, [3 x 3] NumPy array of the instrinsic matrix, [3 x 4] NumPy array of the extrinsic matrix
    """
    ########################
    # TODO: Your code here #
    # Hint: Calculate the rotation matrices for the x, y, and z axes separately.
    # Then multiply them to get the rotational part of the extrinsic matrix.
    ########################
    return initial_camera_matrix_to_replace, initial_intrinsic_matrix_to_replace, initial_extrinsic_matrix_to_replace

def find_coords(camera_matrix):
    """
    This function calculates the coordinates given the student's calculated camera matrix.
    Normalizes the coordinates.
    Already implemented.
    """
    coords = np.matmul(camera_matrix, point_cloud.T)
    return coords / coords[2]


'''
VISUALIZATION
DO NOT ALTER CODE BELOW
'''

# Define initial parameters
init_focalx = 1
init_focaly = 1
init_skew = 0
spacing = 0.05
translation_range = 0.1
rotation_range = 0.5
slider_height = 0.02
slider_width = 0.3
slider_offset_X = 0.1
slider_offset_Y = 0.2

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(figsize=(14,7))
plt.title('Simulated Camera', size=20)
plt.xlim([1, -1])
plt.ylim([1, -1])

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.5)

# Slider for x translation
axtransX = plt.axes([slider_offset_X, 6*spacing + slider_offset_Y, slider_width, slider_height])
transX_slider = Slider(
    ax=axtransX,
    label='x-translation',
    valmin=-translation_range,
    valmax=translation_range,
    valinit=0,
)
# Slider for y translation
axtransY = plt.axes([slider_offset_X, 5*spacing + slider_offset_Y, slider_width, slider_height])
transY_slider = Slider(
    ax=axtransY,
    label='y-translation',
    valmin=-translation_range,
    valmax=translation_range,
    valinit=0,
)
# Slider for z translation
axtransZ = plt.axes([slider_offset_X, 4*spacing + slider_offset_Y, slider_width, slider_height])
transZ_slider = Slider(
    ax=axtransZ,
    label='z-translation',
    valmin=-translation_range,
    valmax=translation_range,
    valinit=0,
)
# Slider for rotation about x axis
axrotX = plt.axes([slider_offset_X, 3*spacing + slider_offset_Y, slider_width, slider_height])
rotX_slider = Slider(
    ax=axrotX,
    label='x-rotation',
    valmin=-rotation_range,
    valmax=rotation_range,
    valinit=0,
)
# Slider for rotation about y axis
axrotY = plt.axes([slider_offset_X, 2*spacing + slider_offset_Y, slider_width, slider_height])
rotY_slider = Slider(
    ax=axrotY,
    label='y-rotation',
    valmin=-rotation_range,
    valmax=rotation_range,
    valinit=0,
)
# Slider for rotation about z axis
axrotZ = plt.axes([slider_offset_X, spacing + slider_offset_Y, slider_width, slider_height])
rotZ_slider = Slider(
    ax=axrotZ,
    label='z-rotation',
    valmin=-rotation_range,
    valmax=rotation_range,
    valinit=0,
)

# Slider for rotation about focal lengths
axfocalX = plt.axes([slider_offset_X, 12*spacing + slider_offset_Y, slider_width, slider_height])
focalX_slider = Slider(
    ax=axfocalX,
    label="x-focal length",
    valmin=0,
    valmax=3,
    valinit=init_focalx,
    orientation="horizontal"
)
axfocalY = plt.axes([slider_offset_X, 11*spacing + slider_offset_Y, slider_width, slider_height])
focalY_slider = Slider(
    ax=axfocalY,
    label="y-focal length",
    valmin=0,
    valmax=3,
    valinit=init_focaly,
    orientation="horizontal"
)

# Slider for rotation about skew
axskew = plt.axes([slider_offset_X, 10*spacing + slider_offset_Y, slider_width, slider_height])
skew_slider = Slider(
    ax=axskew,
    label="skew",
    valmin=-2.5,
    valmax=2.5,
    valinit=init_skew,
    orientation="horizontal"
)
# Sliders for image center coordinates
aximageX = plt.axes([slider_offset_X, 9*spacing + slider_offset_Y, slider_width, slider_height])
imageX_slider = Slider(
    ax=aximageX,
    label="u",
    valmin=-2.5,
    valmax=2.5,
    valinit=0,
    orientation="horizontal"
)
aximageY = plt.axes([slider_offset_X, 8*spacing + slider_offset_Y, slider_width, slider_height])
imageY_slider = Slider(
    ax=aximageY,
    label="v",
    valmin=-2.5,
    valmax=2.5,
    valinit=0,
    orientation="horizontal"
)
# Add labels for sliders
fig.text(0.2, 0.54, "Extrinsic Properties", size="12")
fig.text(0.2, 0.84, "Intrinsic Properties", size="12")

# Add text for intrinsic matrix
kLabelX = -4; kLabelY = -0.65
kLabel = plt.text(kLabelX, kLabelY, "K=", size="12")
kText = [[None,None,None], [None,None,None], [None,None,None]]
for row in range(0,3):
    for col in range(0,3):
        kText[row][col] = plt.text(kLabelX + 0.4 + col*0.5, kLabelY - row*0.10, "{:.2f}".format(0), size="12")

# Add text for extrinsic matrix
rtLabelX = -2; rtLabelY = -0.65
rtLabel = plt.text(rtLabelX, rtLabelY, "Rt=", size="12")
rtText = [[None,None,None,None], [None,None,None,None], [None,None,None,None]]
for row in range(0,3):
    for col in range(0,4):
        rtText[row][col] = plt.text(rtLabelX + 0.4 + col*0.5, rtLabelY - row*0.10, "{:.2f}".format(0), size="12")

# Add text for camera matrix
mLabelX = 0.50; mLabelY = -0.65
mLabel = plt.text(mLabelX, mLabelY, "M=", size="12")
mText = [[None,None,None,None], [None,None,None,None], [None,None,None,None]]
for row in range(0,3):
    for col in range(0,4):
        mText[row][col] = plt.text(mLabelX + 0.4 + col*0.5, mLabelY - row*0.10, "{:.2f}".format(0), size="12")


# list of all sliders
sliders = [transX_slider, transY_slider, transZ_slider, rotX_slider, rotY_slider, rotZ_slider, focalX_slider, focalY_slider, skew_slider, imageX_slider, imageY_slider]

# initial plot calculation using default camera values
camera_matrix, intrinsic_matrix, extrinsic_matrix = calculate_camera_matrix(transX_slider.val, transY_slider.val, transZ_slider.val, rotX_slider.val, rotY_slider.val, rotZ_slider.val, focalX_slider.val, focalY_slider.val, skew_slider.val, imageX_slider.val, imageY_slider.val)
coords = np.matmul(camera_matrix, point_cloud.T)
coords /= coords[2]
plots = ax.plot(coords[0], coords[1],'o', color="blue")[0]

# The function to be called anytime a slider's value changes
def update(val):
    # Replot with new camera properties
    camera_matrix, intrinsic_matrix, extrinsic_matrix = calculate_camera_matrix(transX_slider.val, transY_slider.val, transZ_slider.val, rotX_slider.val, rotY_slider.val, rotZ_slider.val, focalX_slider.val, focalY_slider.val, skew_slider.val, imageX_slider.val, imageY_slider.val)
    coords = find_coords(camera_matrix)
    # Update plotted points in plots
    plots.set_data(coords[0], coords[1])

    # Update text labels for intrinsic matrix
    for row in range(0,3):
        for col in range(0,3):
            kText[row][col].set_text("{:.2f}".format(intrinsic_matrix[row][col]))

    # Update text labels for extrinsic matrix
    for row in range(0,3):
        for col in range(0,4):
            rtText[row][col].set_text("{:.2f}".format(extrinsic_matrix[row][col]))

    # Update text labels
    for row in range(0,3):
        for col in range(0,4):
            mText[row][col].set_text("{:.2f}".format(camera_matrix[row][col]))


# register the update function with each slider
for slider in sliders:
    slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.02, 0.05, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    for slider in sliders:
        slider.reset()
button.on_clicked(reset)

update(None)
plt.show()
