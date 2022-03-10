import edgeiq
import pandas as pd
import os
import cv2
"""
Use pose estimation to determine human poses in realtime. Human Pose returns
a list of key points indicating joints that can be used for applications such
as activity recognition and augmented reality.

To change the engine and accelerator, follow this guide:
https://alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html

To install app dependencies in the runtime container, list them in the
requirements.txt file.
"""

POSES = [
    "Tree_Pose_or_Vrksasana_",
    "Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_",
    "Warrior_I_Pose_or_Virabhadrasana_I_",
    "Warrior_II_Pose_or_Virabhadrasana_II_",
    "Warrior_III_Pose_or_Virabhadrasana_III_"
    ]


def main():
    pose_estimator = edgeiq.PoseEstimation("alwaysai/human_pose")
    pose_estimator.load(engine=edgeiq.Engine.DNN)

    print("Loaded model:\n{}\n".format(pose_estimator.model_id))
    print("Engine: {}".format(pose_estimator.engine))
    print("Accelerator: {}\n".format(pose_estimator.accelerator))

    key_points = [
            'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
            'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee',
            'Right Ankle', 'Left Hip', 'Left Knee', 'Left Ankle']

    header = {}
    for key_point in key_points:
        header['{} x'.format(key_point)] = []
        header['{} y'.format(key_point)] = []

    for pose in POSES:
        df = pd.DataFrame(header)
        print('Generating results for {}'.format(pose))
        image_paths = edgeiq.list_images(os.path.join('images', 'downloads', pose))

        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                results = pose_estimator.estimate(image)
                if len(results.poses) > 0:
                    results = results.poses[0].key_points
                    # Filter only desired key points
                    results = {key: value for key, value in results.items() if key in key_points}
                    new_row = pd.DataFrame(header)
                    for key, value in results.items():
                        if key not in key_points:
                            continue
                        new_row['{} x'.format(key)] = [value[0]]
                        new_row['{} y'.format(key)] = value[1]

                    df = df.append(new_row, ignore_index=True)
                else:
                    print('Skipping {}, no pose detected!'.format(image_path))
            except Exception as e:
                print('Exception on {}! {}'.format(image_path, e))

        df.to_csv('{}.csv'.format(pose))


if __name__ == "__main__":
    main()
