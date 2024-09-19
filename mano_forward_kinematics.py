import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image


class Joint:
    def __init__(self, idx, parent_idx, angle, distance):
        self.idx = idx
        self.parent_idx = parent_idx
        self.angle = angle
        self.distance = distance
        self.test_world_coords = None


class ManoForwardKinematics:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index Finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle Finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring Finger
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky Finger
        ]

    def get_angles(self, direction):
        pitch = np.arcsin(direction[2])
        yaw = np.arctan2(direction[1], direction[0])
        return yaw, pitch

    def normalize_target(self, origin, target):
        vector = target - origin
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def get_distance(self, origin, target):
        return np.linalg.norm(target - origin)

    def calculate_new_position(self, origin, yaw, pitch, distance):
        x0, y0, z0 = origin
        x = x0 + distance * np.cos(pitch) * np.cos(yaw)
        y = y0 + distance * np.cos(pitch) * np.sin(yaw)
        z = z0 + distance * np.sin(pitch)
        return np.array([x, y, z])

    def joint_to_world(self, target_joint, parent_world_pos):
        return self.calculate_new_position(parent_world_pos, target_joint.angle[0], target_joint.angle[1],
                                           target_joint.distance)

    def load_keypoints_from_json(self, input_path):
        all_keypoints = []
        for filename in sorted(os.listdir(input_path)):
            if filename.endswith(".json") and '_0_keypoints_3d' in filename:
                json_path = os.path.join(input_path, filename)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                keypoints = np.array(data).reshape(-1, 3)
                all_keypoints.append(keypoints.tolist())
        return np.array(all_keypoints)

    def store_keypoints_in_json(self, keyframe, keypoints):
        file_name = f'frame_{keyframe:04}_0_keypoints_3d.json'
        output_json_path = os.path.join(self.output_dir, file_name)

        with open(output_json_path, 'w') as json_file:
            json.dump(keypoints, json_file)

        print("Storing Keypoints in JSON")

    def store_joints_in_json(self, keyframe, joints):
        file_name = f'frame_{keyframe:04}_0_joints.json'
        output_json_path = os.path.join(self.output_dir, file_name)

        with open(output_json_path, 'w') as json_file:
            json.dump(joints, json_file)

        print("Storing Joints in JSON")

    def forward_kinetics_add_test_world_coordinates(self, forward_kinematics):
        world_coordinates = {0: (0, 0, 0)}
        for joint_index, joint in forward_kinematics.items():
            joint.test_world_coords = self.joint_to_world(joint, world_coordinates[joint.parent_idx])
            world_coordinates[joint.idx] = self.joint_to_world(joint, world_coordinates[joint.parent_idx])
        root_joint = Joint(0, 0, (0, 0), 0)
        root_joint.test_world_coords = np.array((0, 0, 0))
        forward_kinematics[0] = root_joint

    def draw_plot(self, frame, forward_kinetics):
        fig = plt.figure(figsize=(32, 32))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-.001, .15])
        ax.set_ylim([-.15, .1])
        ax.set_zlim([-.15, .1])
        ax.set_box_aspect([1, 1, 1])

        for joint_index, joint in forward_kinetics.items():
            joint_coord = joint.test_world_coords
            parent_coord = forward_kinetics[joint.parent_idx].test_world_coords
            ax.scatter(joint_coord[0], joint_coord[1], joint_coord[2], color='blue', s=300)
            ax.plot(
                [joint_coord[0], parent_coord[0]],
                [joint_coord[1], parent_coord[1]],
                [joint_coord[2], parent_coord[2]],
                marker='o', linewidth=10, color='red')

        plt.savefig(os.path.join(self.output_dir, f"frame_{frame:04d}.png"))

    def draw_gif(self):
        output_gif_path = os.path.join(self.output_dir, 'hand_kinematic_animation.gif')
        images = [os.path.join(self.output_dir, img) for img in sorted(os.listdir(self.output_dir)) if
                  img.endswith('.png')]
        frames = [Image.open(image) for image in images]
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )

    def process_from_keypoints(self, keypoints):
        count = 0

        for frame in keypoints:
            forward_kinematics = {}
            count += 1
            for connection in self.connections:
                parent_id = connection[0]
                target_id = connection[1]
                origin = np.array(frame[parent_id])
                target = np.array(frame[target_id])

                n_target = self.normalize_target(origin, target)
                angles = self.get_angles(n_target)
                distance = self.get_distance(origin, target)
                forward_kinematics[connection[1]] = Joint(connection[1], parent_id, angles, distance)

            self.forward_kinetics_add_test_world_coordinates(forward_kinematics)
            self.draw_plot(count, forward_kinematics)
            self.store_keypoints(count, forward_kinematics)
            self.store_joints(count, forward_kinematics)
        self.draw_gif()

    def store_keypoints(self, keyframe, forward_kinematics):
        keypoints = []
        for joint_index, joint in forward_kinematics.items():
            keypoints.append(joint.test_world_coords.tolist())
        self.store_keypoints_in_json(keyframe, keypoints)

    def store_joints(self, keyframe, forward_kinematics):
        joints = []
        for joint_index, joint in forward_kinematics.items():
            joints.append({'index:': joint.idx, "parent_index": joint.parent_idx, "distance": joint.distance, "angle": joint.angle})
        self.store_joints_in_json(keyframe, joints)

    def process_from_path(self, path):
        all_keypoints_np = self.load_keypoints_from_json(path)
        self.process_from_keypoints(all_keypoints_np)


print("Starting process")
output_dir = "./video/kinematic_animation"
mano_fk = ManoForwardKinematics(output_dir)
mano_fk.process_from_path("./video/video_out")
# mano_fk.process_from_keypoints()
print("Done")
