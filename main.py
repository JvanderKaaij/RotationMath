import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def main():
    # Define two points in 3D space
    point = np.array([0, 0, 0])
    point2 = np.array([-1, 0, 1])

    # Use the point2 vector (not point) as the direction, normalize it
    direction = point2 / np.linalg.norm(point2)

    # Define the axes
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # Calculate the angles between the direction and the axes using dot product
    theta_x = np.arccos(np.dot(x_axis, direction))
    theta_y = np.arccos(np.dot(y_axis, direction))
    theta_z = np.arccos(np.dot(z_axis, direction))

    # Convert the angles to degrees
    print("Theta_x (degrees):", np.degrees(theta_x))
    print("Theta_y (degrees):", np.degrees(theta_y))
    print("Theta_z (degrees):", np.degrees(theta_z))

    # # Get quaternion from axis-angle
    # if np.linalg.norm(rotation_vector) > 0:  # Avoid division by zero if vectors are parallel
    #     rotation_vector = rotation_vector / np.linalg.norm(rotation_vector)  # Normalize
    #     quaternion = R.from_rotvec(rotation_vector * angle).as_quat()  # Axis-angle to quaternion
    # else:
    #     quaternion = np.array([0, 0, 0, 1])  # If vectors are already aligned


    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the point
    ax.scatter(point[0], point[1], point[2], color='red', s=100)  # s is the size of the point
    ax.scatter(point2[0], point2[1], point2[2], color='blue', s=100)  # s is the size of the point

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set the limits of the axes
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    plt.show()


main()