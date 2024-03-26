import logging as log
import os
import shutil
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import tensorflow as tf
import PySide6
from PySide6 import *
from PySide6.QtCore import QSize, Qt, QFile
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QApplication, QPlainTextEdit
from PySide6.QtUiTools import QUiLoader
from tensorflow.keras.layers import Dropout, Dense, LSTM

from helpers import resolution
from images_capture import open_images_capture
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter
from openvino.model_zoo.model_api.models import ImageModel, OutputTransform
from openvino.model_zoo.model_api.performance_metrics import PerformanceMetrics
from openvino.model_zoo.model_api.pipelines import get_user_config, AsyncPipeline

import threading
import time

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

plugin_path = os.path.join(os.path.dirname(PySide6.__file__), 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

ARCHITECTURES = {
    'ae': 'HPE-assosiative-embedding',
    'higherhrnet': 'HPE-assosiative-embedding',
    'openpose': 'openpose'
}

input_shape2 = (5, 17 * 2)
num_classes2 = 2
source_path2 = './source'
tag2 = [[0], [1]]
tag_class2 = {'timber': 0, 'walk': 1}

"""
需要设置的有：
checkpoint文件夹（可使用LSTM_20220614.py训练）
fps_extract（需要与checkpoint训练所用数据抽取的帧率相匹配）
ycl()函数（需要与checkpoint训练所用数据的输入格式相匹配）
"""
fps_extract = 2  # 抽取的帧率

"""
以下设置运行参数
"""
model1 = r'./intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml'
input1 = 'VIDEO/xck/timber_x_1.avi'
architecture_type1 = 'openpose'
output_limit1 = 1000
device1 = 'CPU'
prob_threshold1 = 0.1
tsize1 = None
layout1 = None
utilization_monitors1 = ''
num_infer_requests1 = 0
num_threads1 = None
no_show1 = False
output_resolution1 = None
raw_output_message1 = False
loop1 = False
output1 = None
num_streams1 = ''
output_tag = {0: '摔倒', 1: '正常'}
thread_lock = threading.Lock()
# 用来存放解析线程
g_parse_list = []
msg_temp = ''


# Define a simple sequential model
# return_sequences一般在最后一层为False
def create_model_lstm1():
    model = tf.keras.Sequential([
        LSTM(300, return_sequences=True, input_shape=input_shape2),
        Dropout(0.1),
        LSTM(100),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'], )
    return model


def change_shape(arr=None):
    if arr is None:
        arr = []
    arr_changed = np.ones((5, 17 * 2))
    for i in range(0, 17):
        for j in range(0, 2):
            arr_changed[0][i * 2 + j] = arr[0][i][j]
            arr_changed[1][i * 2 + j] = arr[1][i][j]
            arr_changed[2][i * 2 + j] = arr[2][i][j]
            arr_changed[3][i * 2 + j] = arr[3][i][j]
            arr_changed[4][i * 2 + j] = arr[4][i][j]
    return arr_changed


"""
在下方更改对每一帧的数据的预处理方式
预处理的输入参数数组shape为(17, 2)
"""


# def ycl(a):
#     a1 = a[:, 0]
#     a2 = a[:, 1]
#     b1 = a1.nonzero()
#     c1 = a1[b1].min()
#     b2 = a2.nonzero()
#     c2 = a2[b2].min()
#     max = a2[b2].max()
#     z = max - c2
#     a1 = a1 - c1
#     a2 = a2 - c2
#     a1 = a1 / z * 100
#     a2 = a2 / z * 100
#     c = np.zeros((17, 2))
#     c[:, 0] = a1
#     c[:, 1] = a2
#     c = abs(c)
#     return c


def ycl(a):
    a1 = a[:, 0]
    a2 = a[:, 1]
    b1 = a1.nonzero()
    c1 = a1[b1].min()
    b2 = a2.nonzero()
    c2 = a2[b2].min()
    max1 = a2[b2].max()
    max2 = a1[b1].max()
    z = max(max1 - c1, max2 - c2)
    a1 = a1 - c1
    a2 = a2 - c2
    a1 = a1 / z * 10000
    a2 = a2 / z * 10000
    c = np.zeros((17, 2))
    c[:, 0] = a1
    c[:, 1] = a2
    c = abs(c)
    c = np.around(c, 3)
    return c


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                      type=str, required=True, choices=('ae', 'higherhrnet', 'openpose'))
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('-t', '--prob_threshold', default=0.1, type=float,
                                   help='Optional. Probability threshold for poses filtering.')
    common_model_args.add_argument('--tsize', default=None, type=int,
                                   help='Optional. Target input size. This demo implements image pre-processing '
                                        'pipeline that is common to human pose estimation approaches. Image is first '
                                        'resized to some target size and then the network is reshaped to fit the input '
                                        'image shape. By default target image size is determined based on the input '
                                        'shape from IR. Alternatively it can be manually set via this parameter. Note '
                                        'that for OpenPose-like nets image is resized to a predefined height, which is '
                                        'the target size in this case. For Associative Embedding-like nets target size '
                                        'is the length of a short first image side.')
    common_model_args.add_argument('--layout', type=str, default=None,
                                   help='Optional. Model inputs layouts. '
                                        'Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=0, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('-no_show', '--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


def txt_save(filename, data):
    print(data)
    print(type(data))
    file = open(filename, 'a')
    for i in range(len(data)):
        s = data[i]
        file.write(s)
    file.write('\r\n')
    file.close()


def npy_save(filename, data):
    np.save(filename, data)


# 建立五个对象得到临时点数据
object_tem = np.zeros((5, 17, 2))
object_tem[0] = np.zeros((17, 2))
object_tem[1] = np.zeros((17, 2))


object_tem[2] = np.zeros((17, 2))
object_tem[3] = np.zeros((17, 2))
object_tem[4] = np.zeros((17, 2))


def get_score(point):
    score = 0
    # point = point.flatten()
    # for i in point:
    #     score += abs(i)
    score = point[0][0]+point[0][1]
    return score


def get_score2(point):
    score = 0
    # point = point.flatten()
    # for i in point:
    #     score += abs(i)
    score = point[16][0]+point[16][1]
    print(point)
    # print(score)
    return score


def save_obj_frame(point, number):
    global object_tem
    score = [0, 0]
    point_score = get_score(point)
    score[0] = get_score(object_tem[0])
    score[1] = get_score(object_tem[1])
    for i in range(2):
        score[i] -= point_score
        score[i] = abs(score[i])
    if number == 1:
        score_tag = score.index(min(score))
        object_tem[score_tag] = point
        return score_tag
    else:
        score_tag = score.index(min(score))
        object_tem[score_tag] = point
        return score_tag


def get_obj(point):
    global object_tem
    score = [0, 0]
    point_score = get_score(point)
    score[0] = get_score(object_tem[0])
    score[1] = get_score(object_tem[1])
    for i in range(2):
        if point_score == score[i]:
            return i
    return 0


def save_obj_frame2(point):
    global object_tem
    score = [0, 0, 0, 0, 0]
    point_score = get_score(point)
    score[0] = get_score(object_tem[0])
    score[1] = get_score(object_tem[1])
    score[2] = get_score(object_tem[2])
    score[3] = get_score(object_tem[3])
    score[4] = get_score(object_tem[4])
    for i in range(5):
        score[i] -= point_score
    for i in range(5):
        score[i] = abs(score[i])
    if get_score(object_tem[0]) == 0:
        object_tem[0] = point
    elif get_score(object_tem[1]) == 0:
        object_tem[1] = point
    elif get_score(object_tem[2]) == 0:
        object_tem[2] = point
    elif get_score(object_tem[3]) == 0:
        object_tem[3] = point
    elif get_score(object_tem[4]) == 0:
        object_tem[4] = point
    score_tag = score.index(min(score))
    x = 1680
    if score[score_tag] < x:
        for i in range(5):
            if (object_tem[i] == point).all():
                object_tem[i] = np.zeros((17, 2))
        object_tem[score_tag] = point
        return score_tag
    else:
        return 0


def get_obj2(point):
    global object_tem
    score = [0, 0, 0, 0, 0]
    point_score = get_score(point)
    score[0] = get_score(object_tem[0])
    score[1] = get_score(object_tem[1])
    score[2] = get_score(object_tem[2])
    score[3] = get_score(object_tem[3])
    score[4] = get_score(object_tem[4])
    for i in range(5):
        if point_score == score[i]:
            return i
    return 0


default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
                    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
    (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
    (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
    (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
    (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
    (0, 170, 255))


def draw_poses(img, poses, point_score_threshold, output_transform, skeleton=default_skeleton, draw_ellipses=False):
    img = output_transform.resize(img)
    if poses.size == 0:
        return img
    stick_width = 4
    pose_number = poses.size / 3 /17
    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points = output_transform.scale(points)
        # print('*'*20)
        # print('1:')
        # print(points)
        # txt_save('1234.txt',points)
        points_scores = pose[:, 2]
        # txt_save('123.txt', points)
        # Draw joints.
        pose_num = save_obj_frame(points, pose_number)
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)

        # 调用cv2.putText()添加文字
        # text = "Object{}".format(pose_num)
        # cv2.putText(img, text, (points[0][0], points[0][1] - 70), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)

        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                               angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img, points


def get_points(pose, output_transform):
    if pose.size == 0:
        return np.zeros((17, 2))
    points = pose[:, :2].astype(np.int32)
    points = output_transform.scale(points)
    # txt_save('1234.txt',points)
    # points_scores = pose[:, 2]
    return points


def data_generate(data_1, data_2):
    for i in range(0, 4):
        data_1[i] = data_1[i + 1]
    data_1[4] = data_2[0]
    return data_1


class MyThread(threading.Thread):
    def __init__(self, thread_id):
        super(MyThread, self).__init__()
        self.thread_id = thread_id
        self.name = "Thread-%d" % self.thread_id
        self.ret = True
        # 计数，前5次不判断
        self.counter = 0
        # 存points
        self.data_lstm = np.zeros((5, 17, 2))
        self.State = '正常'
        self.state_msg = "对象{}'s Current State: 待检测, time: {}".format(self.thread_id, time.asctime(time.localtime()))

    def get_state(self, points, model):
        points2 = ycl(points)
        points2 = np.reshape(points2, (1, 17, 2))
        # time_start = time.time()  # 开始计时
        data_lstm = data_generate(self.data_lstm, points2)
        data_reshape = change_shape(data_lstm)
        data_reshape = np.reshape(data_reshape, (1, 5, 17 * 2))
        result_lstm = model.predict([data_reshape])
        predict_tag = tf.argmax(result_lstm, axis=1)
        predict_tag = int(predict_tag)
        self.State = output_tag[predict_tag]
        if self.counter < 5:
            self.State = '待检测'
            predict_tag = 1
        self.counter += 1
        self.state_msg = "当前状态: {}".format(self.State)
        # time.strftime('%H:%M:%S')
        time1 = time.strftime('%H:%M:%S')
        msg2 = "发现摔倒，时间: {}".format( time1)
        # time_end = time.time()  # 结束计时
        # time_c = time_end - time_start  # 运行所花时间
        # print('time cost: %.3fs' % time_c)
        return self.state_msg, predict_tag, msg2


def print_raw_results(poses, scores, frame_id):
    log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
    for pose, pose_score in zip(poses, scores):
        pose_str = ' '.join('({:.2f}, {:.2f}, {:.2f})'.format(p[0], p[1], p[2]) for p in pose)
        log.debug('{} | {:.2f}'.format(pose_str, pose_score))


class Window:
    def __init__(self):
        super(Window, self).__init__()
        # 从ui文件中加载UI定义
        # qfile = QFile("FallDetectUI.ui")
        qfile = QFile("FallDetectUI.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        self.ui = QUiLoader().load(qfile)
        # 从UI定义中动态创建一个相应的窗口对象
        self.video_size = QSize(640, 360)
        self.ui.setWindowTitle('Fall Detection System')
        # self.image_label = QLabel()
        self.ui.VideoLabel.setFixedSize(self.video_size)
        self.ui.VideoLabel.setScaledContents(True)

    def display_video_stream(self, frame_stream):
        """Read frame from camera and repaint QLabel widget.
        """
        frame_stream = cv2.cvtColor(frame_stream, cv2.COLOR_RGB2BGR)
        image = QImage(frame_stream, frame_stream.shape[1], frame_stream.shape[0],
                       frame_stream.strides[0], QImage.Format_RGB888)
        self.ui.VideoLabel.setPixmap(QPixmap.fromImage(image))
        self.ui.show()
        # sys.exit(0)


def main():
    app = QApplication([])
    cap = open_images_capture(input1, loop1)
    next_frame_id = 1
    next_frame_id_to_show = 0

    metrics = PerformanceMetrics()
    render_metrics = PerformanceMetrics()
    video_writer = cv2.VideoWriter()

    plugin_config = get_user_config(device1, num_streams1, num_threads1)
    model_adapter = OpenvinoAdapter(create_core(), model1, device=device1, plugin_config=plugin_config,
                                    max_num_requests=num_infer_requests1,
                                    model_parameters={'input_layouts': layout1})

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    config = {
        'target_size': tsize1,
        'aspect_ratio': frame.shape[1] / frame.shape[0],
        'confidence_threshold': prob_threshold1,
        'padding_mode': 'center' if architecture_type1 == 'higherhrnet' else None,
        # the 'higherhrnet' and 'ae' specific
        'delta': 0.5 if architecture_type1 == 'higherhrnet' else None,  # the 'higherhrnet' and 'ae' specific
    }
    model = ImageModel.create_model(ARCHITECTURES[architecture_type1], model_adapter, config)
    model.log_layers_info()

    hpe_pipeline = AsyncPipeline(model)
    hpe_pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})

    output_transform = OutputTransform(frame.shape[:2], output_resolution1)
    if output_resolution1:
        output_resolution = output_transform.new_resolution
    else:
        output_resolution = (frame.shape[1], frame.shape[0])
    # presenter = monitors.Presenter(utilization_monitors1, 55,
    #                              (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
    if output1 and not video_writer.open(output1, cv2.VideoWriter_fourcc(*'MJPG'), cap.fps(),
                                         output_resolution):
        raise RuntimeError("Can't open video writer")
    i = 0
    # shutil.rmtree(r'./npy')
    # os.mkdir(r'./npy')
    # Create a basic model instance
    model_1 = create_model_lstm1()
    checkpoint_save_path = "./checkpoint/lstm.ckpt"
    model_1.load_weights(checkpoint_save_path)

    thread_num = 2
    # 创建n个线程,输入：points（17, 2），输出是此数据的状态
    for i in range(thread_num):
        # 创建一个解析线程Thread-i
        tparse = MyThread(i)
        tparse.start()
        # 保存到列表中
        g_parse_list.append(tparse)
    space_counter = 0

    win = Window()
    win.ui.show()
    while True:
        if hpe_pipeline.callback_exceptions:
            raise hpe_pipeline.callback_exceptions[0]
        # Process all completed requests
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        if results:
            (poses, scores), frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']
            if len(poses) and raw_output_message1:
                print_raw_results(poses, scores, next_frame_id_to_show)

            # presenter.drawGraphs(frame)
            rendering_start_time = perf_counter()
            # frame, points = draw_poses(frame, poses, prob_threshold1, output_transform)
            if not len(poses) == 0:
                frame, points = draw_poses(frame, poses, prob_threshold1, output_transform)
            render_metrics.update(rendering_start_time)
            metrics.update(start_time, frame)
            if video_writer.isOpened() and (output_limit1 <= 0 or next_frame_id_to_show <= output_limit1 - 1):
                video_writer.write(frame)
            next_frame_id_to_show += 1
            frame_stream = frame
            win.display_video_stream(frame_stream)
            output_transform2 = output_transform
            space_counter += 1
            if space_counter % (fps_extract + 1) == 0:
                obj = 0
                if obj == len(poses) / 3 / 17:
                    obj = 0
                for pose in poses:
                    points_obj = get_points(pose, output_transform2)
                    # print('2:')
                    # print(points_obj)
                    obj = get_obj(points_obj)
                    # print(obj)
                    msg, tag, msg2 = g_parse_list[obj].get_state(points_obj, model_1)
                    # win.ui.CurrentStateText.append(msg)
                    global msg_temp
                    current_state = ['当前状态: 正常', '当前状态: 有人摔倒']
                    if tag == 0 and msg_temp != msg2:
                        # win.ui.FallTimeText.setText(msg)
                        win.ui.CurrentStateText.clear()
                        obj_number = '当前有{}人'.format(int(poses.size / 3 / 17))
                        win.ui.CurrentStateText.append(obj_number)
                        win.ui.CurrentStateText.append(current_state[1])
                        win.ui.FallTimeText.append(msg2)
                        msg_temp = msg2
                    else:
                        win.ui.CurrentStateText.clear()
                        obj_number = '当前有: {}人'.format(int(poses.size / 3 / 17))
                        win.ui.CurrentStateText.append(obj_number)
                        win.ui.CurrentStateText.append(current_state[0])

            # win.ui.CurrentState.setText(output_tag[predict_tag])

            if not no_show1:
                key = cv2.waitKey(1)
                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                # presenter.handleKey(key)
            continue

        if hpe_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                break

            # Submit for inference
            hpe_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            # Wait for empty request
            hpe_pipeline.await_any()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
