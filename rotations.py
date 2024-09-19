import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

# Define MANO hand keypoint labels and connections
mano_labels = [
    "Wrist", "Thumb_MCP", "Thumb_IP", "Thumb_DIP", "Thumb_Tip",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip"
]

# Define connections for MANO hand keypoints
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index Finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle Finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring Finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky Finger
]

# Path to the directory containing JSON files
input_dir = "./video/kinematic_animation"

# List to hold all keypoints from selected files
all_keypoints = []

def calculate_local_axes(keypoints):
    """
    Calculate the local axes of the hand based on keypoints.

    Args:
        keypoints (np.ndarray): An array of shape (N, 3) representing the keypoints of the object.

    Returns:
        np.ndarray: A 3x3 matrix representing the local X, Y, Z axes of the object.
    """
    # Reference point (wrist position)
    wrist_pos = keypoints[0]

    # Define Z-axis: vector from wrist (0) to middle finger MCP (9)
    z_axis = keypoints[9] - wrist_pos
    z_axis /= np.linalg.norm(z_axis)

    # Define X-axis: vector from index MCP (5) to ring MCP (13)
    x_axis = keypoints[13] - keypoints[5]
    # Make x_axis orthogonal to z_axis
    x_axis -= z_axis * np.dot(x_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Define Y-axis as cross product of z-axis and x-axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Construct local rotation matrix
    local_axes = np.stack([x_axis, y_axis, z_axis], axis=1)
    return local_axes

def calculate_rotation_angles(rotation_matrix):
    """
    Calculate the rotation angles around X, Y, and Z axes from a rotation matrix.

    Args:
        rotation_matrix (np.ndarray): 3x3 rotation matrix.

    Returns:
        tuple: Rotation angles (x_angle, y_angle, z_angle) in degrees.
    """
    x_angle = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
    y_angle = np.degrees(np.arctan2(-rotation_matrix[2, 0],
                                    np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)))
    z_angle = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))

    return x_angle, y_angle, z_angle

def create_inverse_rotation_matrix(x_angle):
    """
    Create an inverse rotation matrix from a given X rotation angle, setting Y and Z angles to 0.

    Args:
        x_angle (float): Rotation around the X-axis in degrees.

    Returns:
        np.ndarray: 3x3 inverse rotation matrix.
    """
    # Convert the X-angle to radians
    x_rad = np.deg2rad(-x_angle)

    # Create the rotation matrix for X-axis only
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(x_rad), -np.sin(x_rad)],
        [0, np.sin(x_rad), np.cos(x_rad)]
    ])

    # Since Y and Z rotations are set to 0, we return only the X rotation matrix
    return R_x


def apply_dynamic_rotation(keypoints, axis='y', angle=90):
    """
    Apply a rotation around the specified axis to the keypoints.

    Args:
        keypoints (np.ndarray): An array of shape (N, 3) representing the keypoints of the object.
        axis (str): The axis around which to rotate ('x', 'y', or 'z').
        angle (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The rotated keypoints after applying the specified rotation.
    """
    # Convert angle to radians
    rad = np.deg2rad(angle)

    # Create the appropriate rotation matrix based on the specified axis
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # Apply the rotation to the keypoints
    rotated_keypoints = (keypoints - keypoints[0]) @ R.T + keypoints[0]
    return rotated_keypoints

def align_view_to_palm(keypoints):
    """
    Align the view to the palm of the hand by calculating the rotation required
    to keep the palm facing the viewer.

    Args:
        keypoints (np.ndarray): An array of shape (N, 3) representing the keypoints of the hand.

    Returns:
        np.ndarray: Rotated keypoints aligned to the palm view.
    """
    # Reference points: wrist (0), middle finger MCP (9), and index finger MCP (5)
    wrist = keypoints[0]
    middle_mcp = keypoints[9]
    index_mcp = keypoints[5]

    # Define palm normal vector using cross product between two vectors on the palm plane
    palm_normal = np.cross(middle_mcp - wrist, index_mcp - wrist)
    palm_normal /= np.linalg.norm(palm_normal)

    # Desired palm normal facing the camera (Z-axis direction)
    desired_normal = np.array([0, 0, 1])

    # Calculate rotation axis and angle to align palm normal with desired direction
    rotation_axis = np.cross(palm_normal, desired_normal)
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.clip(np.dot(palm_normal, desired_normal), -1.0, 1.0))

    # Create rotation matrix to align palm to the desired view
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * K @ K

    # Rotate keypoints to align the palm view
    aligned_keypoints = (keypoints - wrist) @ rotation_matrix.T + wrist

    return aligned_keypoints

def calculate_object_center(keypoints):
    """
    Calculate the center of the object based on the keypoints.

    Args:
        keypoints (np.ndarray): Keypoints of the object.

    Returns:
        np.ndarray: The center of the object.
    """
    return np.mean(keypoints, axis=0)

# Function to update each frame in the animation
def update(frame):
    """
    Update function to adjust each frame of the hand keypoint animation.

    Args:
        frame (int): The current frame index.
    """
    # Clear the plots
    ax.clear()
    ax2.clear()

    # Set axis labels for both subplots
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Get keypoints for the current frame
    keypoints = all_keypoints_np[frame]

    # Calculate the local axes based on the keypoints
    local_axes = calculate_local_axes(keypoints)

    # Get the rotation angles (only X will be used, Y and Z will be set to 0)
    rotation_angles = calculate_rotation_angles(local_axes)

    print(f"Local Rotation Angles: {rotation_angles}")

    # Set Y and Z rotation angles to 0 degrees
    adjusted_rotation_angles = (rotation_angles[0], 0, 0)

    # Create the inverse rotation matrix using only the X angle
    inverse_rotation_matrix = create_inverse_rotation_matrix(adjusted_rotation_angles[0])

    # Apply the inverse rotation to align the hand with the camera's coordinate system
    aligned_keypoints = (keypoints - keypoints[0]) @ inverse_rotation_matrix.T + keypoints[0]

    # Align the view to the palm of the hand
    manipulated_keypoints = align_view_to_palm(aligned_keypoints)
    manipulated_axes = calculate_local_axes(manipulated_keypoints)
    manipulated_rotation_angles = calculate_rotation_angles(manipulated_axes)

    # Plot original (aligned) keypoints on the left subplot
    ax.scatter(aligned_keypoints[:, 0], aligned_keypoints[:, 1], aligned_keypoints[:, 2], c='r', s=50)
    for start, end in connections:
        ax.plot(
            [aligned_keypoints[start, 0], aligned_keypoints[end, 0]],
            [aligned_keypoints[start, 1], aligned_keypoints[end, 1]],
            [aligned_keypoints[start, 2], aligned_keypoints[end, 2]],
            c='b'
        )

    # Plot manipulated (aligned to palm) keypoints on the right subplot
    ax2.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='g', s=50)
    for start, end in connections:
        ax2.plot(
            [keypoints[start, 0], keypoints[end, 0]],
            [keypoints[start, 1], keypoints[end, 1]],
            [keypoints[start, 2], keypoints[end, 2]],
            c='b'
        )

    # Annotate both subplots with translation and rotation angles
    object_center = calculate_object_center(aligned_keypoints)
    ax.text2D(0.05, 0.95, f"Translation: {object_center}", transform=ax.transAxes, fontsize=10, color='green')
    ax.text2D(
        0.05, 0.90,
        f"Rotation X: {rotation_angles[0]:.2f}°, Y: 0.00°, Z: 0.00°",
        transform=ax.transAxes, fontsize=10, color='blue'
    )

    # Annotate the manipulated plot with the rotation angles after palm alignment
    ax2.text2D(0.05, 0.95, f"Palm Aligned Rotation:", transform=ax2.transAxes, fontsize=10, color='green')
    ax2.text2D(
        0.05, 0.90,
        f"Rotation X: {manipulated_rotation_angles[0]:.2f}°, Y: {manipulated_rotation_angles[1]:.2f}°, Z: {manipulated_rotation_angles[2]:.2f}°",
        transform=ax2.transAxes, fontsize=10, color='blue'
    )

    # Save each frame as an image
    plt.savefig(os.path.join(output_dir, f"frame_{frame:04d}.png"))


# STARTS FROM HERE -------------------------------

print("Start")

# Iterate over each JSON file in the input directory and extract keypoints
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".json") and '_0_keypoints_3d' in filename:
        json_path = os.path.join(input_dir, filename)

        # Load the JSON file containing keypoints
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract keypoints as a NumPy array and reshape
        keypoints = np.array(data).reshape(-1, 3)

        # Append the keypoints to the list
        all_keypoints.append(keypoints.tolist())

# Convert the list of keypoints into a NumPy array for plotting
all_keypoints_np = np.array(all_keypoints)

# Create a figure for the animation
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(121, projection='3d')  # Original result on the left
ax2 = fig.add_subplot(122, projection='3d')  # Manipulated result on the right

# Set plot labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Output directory for saving images
output_dir = "./video/animation_images"
os.makedirs(output_dir, exist_ok=True)

# Manually save each frame using the update function
for frame in range(len(all_keypoints_np)):
    update(frame)

# Create GIF from saved images
output_gif_path = os.path.join(output_dir, 'hand_keypoints_animation_before_after_rotation.gif')

# Collect all the saved images in the output directory
images = [os.path.join(output_dir, img) for img in sorted(os.listdir(output_dir)) if img.endswith('.png')]

# Load images and create GIF
frames = [Image.open(image) for image in images]

# Save frames as a GIF
frames[0].save(
    output_gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=100,  # Duration of each frame in milliseconds
    loop=0  # Loop forever
)

print(f"GIF saved at {output_gif_path}.")

# Save keypoints to a JSON file
output_json_path = os.path.join(output_dir, 'keypoints_keyframes_corrected.json')
with open(output_json_path, 'w') as f:
    json.dump(all_keypoints, f, indent=4)

print(f"Keyframes saved at {output_json_path}.")
