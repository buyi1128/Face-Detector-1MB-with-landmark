from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
from data import cfg_mnet, cfg_slim, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm

from track import Tracker
# from utils.timer import Timer
from vision.utils.misc import Timer

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/slim_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='slim', help='Backbone network mobile0.25 or slim or RFB')
parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--long_side', default=320, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.1, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=100, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.1, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=20, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.2, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def detect_face(net, img):

    img = np.float32(img_raw)
    # testing scale
    target_size = args.long_side
    max_size = args.long_side
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if args.origin_size:
        resize = 1

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        # print("\nimg shape resize: ", img.shape)
    im_height, im_width, _ = img.shape

    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    # print('net forward time: {:.4f}'.format(time.time() - tic))


    tic = time.time()

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    # print('post processing time: {:.4f}'.format(time.time() - tic))
    return dets



if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    net = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg = cfg, phase = 'test')
    elif args.network == "slim":
        cfg = cfg_slim
        net = Slim(cfg = cfg, phase = 'test')
    elif args.network == "RFB":
        cfg = cfg_rfb
        net = RFB(cfg = cfg, phase = 'test')
    else:
        print("Don't support network!")
        exit(0)

    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    ## choose input source
    source_code = 3  # 1 from camera, 2 from computer camera, 3 from video file
    if source_code == 1:
        url = "rtsp://admin:12345678wwas@192.168.0.7:554/"
        cap = cv2.VideoCapture(url)
    elif source_code == 2:
        cap = cv2.VideoCapture(0)  # capture from camera
    elif source_code == 3:
        video_path = "./videos/first.mp4"
        cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_FPS, 15)

    # people = None
    preframe_num = 5
    tracker = Tracker(frame_num=preframe_num)
    current_lable = None
    index = 0
    count = 0
    while True:
        ret, img_raw = cap.read()
        if img_raw is None:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        dets = detect_face(net, img_raw)
        if len(dets) == 0:
            continue
        count += 1
        boxes = [x[0:4] for x in dets]
        print("\n *******  count: ", count, "   ***************")
        print("boxes: ", len(boxes), boxes[0])
        if current_lable is None:
                current_lable = tracker.track(boxes)
                for i in range(len(current_lable)):
                    current_lable[i] = i
                index = len(current_lable)
        else:
            current_lable = tracker.track(boxes)
            for i in range(len(current_lable)):
                if current_lable[i] == -1:
                    current_lable[i] = index + 1
                    index += 1
        print("curent_label: ", len(current_lable), current_lable)
        print("index: ", index)

        tracker.update(boxes, current_lable)
        # show image
        if args.save_image:
            tic = time.time()
            for i, b in enumerate(dets):
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cx = int((b[0] + b[2]) / 2)
                cy = int(b[3] - (b[3]-b[1])/10)
                cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255))
                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

                ## track
                cx = int((b[0] + b[2]) / 2)
                cy = int(b[1] + (b[3] - b[1]) / 8)
                cv2.putText(img_raw,str(current_lable[i]), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255))

            # save image
            # name = "test.jpg"
            # cv2.imwrite(name, img_raw)
            print('draw circle img time: {:.4f}'.format(time.time() - tic))
            if cv2.waitKey(1) & 0xFF == ord('p'):
                cv2.waitKey()
            cv2.imshow("frame", img_raw)



'''
   track = Tracker(frame_num=2)

    label_1 = track.track(pre_frames_1)                                                                                                                                                    
    for i in range(len(label_1)):
        label_1[i] = i

    track.update(pre_frames_1, label_1)

    label_2 = track.track(pre_frames_2)
    print(label_2)
    track.update(pre_frames_2, label_2)
    label_3 = track.track(frame)

    print(label_1, label_2, label_3)
'''