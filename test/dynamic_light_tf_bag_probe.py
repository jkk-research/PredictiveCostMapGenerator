import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def main(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    types = {topic.name: get_message(topic.type) for topic in reader.get_all_topics_and_types()}
    frames = {}
    counts = {}
    transforms = []
    egos = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in ('/tf', '/tf_static', '/ego', '/scenario', '/plan/trajectory'):
            continue
        message = deserialize_message(data, types[topic])
        counts[topic] = counts.get(topic, 0) + 1
        if topic in ('/tf', '/tf_static'):
            for transform in message.transforms:
                key = (transform.header.frame_id, transform.child_frame_id)
                frames[key] = frames.get(key, 0) + 1
                transforms.append((transform, topic))
        elif topic == '/ego':
            frames[('ego.header.frame_id', message.header.frame_id)] = 1
            egos.append(message)
        elif topic == '/scenario':
            frames[('scenario.header.frame_id', message.header.frame_id)] = 1
        elif topic == '/plan/trajectory':
            frames[('trajectory.header.frame_id', message.header.frame_id)] = 1
    print('topic counts:', counts)
    print('frame pairs and counts:')
    for pair, count in sorted(frames.items()):
        print(pair, count)
    ego_transforms = [(transform, topic) for transform, topic in transforms
                      if transform.child_frame_id == 'base_link']
    if ego_transforms and egos:
        offsets = []
        for transform, _ in ego_transforms:
            ego = min(egos, key=lambda message: abs(
                message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec -
                transform.header.stamp.sec * 1_000_000_000 - transform.header.stamp.nanosec))
            error = ((transform.transform.translation.x - ego.pose.pose.position.x)**2 +
                     (transform.transform.translation.y - ego.pose.pose.position.y)**2)**0.5
            offsets.append((abs(
                ego.header.stamp.sec * 1_000_000_000 + ego.header.stamp.nanosec -
                transform.header.stamp.sec * 1_000_000_000 - transform.header.stamp.nanosec), error))
        print('TF/Ego max stamp delta ms:', max(value[0] for value in offsets) / 1e6)
        print('TF/Ego max XY delta m:', max(value[1] for value in offsets))
        print('first dynamic TF stamp/frame:', ego_transforms[0][0].header.stamp,
              ego_transforms[0][0].header.frame_id, ego_transforms[0][0].child_frame_id)
        print('last dynamic TF stamp/frame:', ego_transforms[-1][0].header.stamp,
              ego_transforms[-1][0].header.frame_id, ego_transforms[-1][0].child_frame_id)


if __name__ == '__main__':
    main(sys.argv[1])