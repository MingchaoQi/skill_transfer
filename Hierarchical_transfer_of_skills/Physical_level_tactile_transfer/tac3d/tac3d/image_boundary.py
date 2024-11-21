import sys, getopt
import numpy as np
import cv2
import os
from gelsight import gsdevice
import gs3drecon

def get_diff_img(img1, img2):
    """Calculate the difference between two images."""
    return np.clip((img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)

def get_diff_img_2(img1, img2):
    """Normalize the difference between two images into the range 0.5 around zero."""
    return (img1 * 1.0 - img2) / 255. + 0.5

def choose_boundary(depth_img, img_bin, bound):
    """Filter and enhance the depth image based on depth boundaries."""
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            if bound[0] < depth_img[i, j] < bound[1]:
                img_bin[int(i * 2), int(j * 2)] = 255
    return img_bin

def boundary_depth(cam_id):
    """Initialize camera and load neural network for 3D reconstruction."""
    dev = gsdevice.Camera(cam_id)
    net_file_path = 'src/tac3d/nnmini.pt'
    model_file_path = '.'
    gpuorcpu = "cuda" if GPU else "cpu"
    dev.connect()
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)
    print("Current working directory:", os.getcwd())
    nn = gs3drecon.Reconstruction3D(dev)
    net = nn.load_nn(net_path, gpuorcpu)
    return nn, dev

if __name__ == "__main__":
    argv = sys.argv[1:]
    device = "mini"
    try:
        opts, args = getopt.getopt(argv, "hd:", ["device="])
    except getopt.GetoptError:
        print('Usage: python show3d.py -d <device>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: show3d.py -d <device>')
            sys.exit()
        elif opt in ("-d", "--device"):
            device = arg

    GPU = False
    SAVE_VIDEO_FLAG = False
    MASK_MARKERS_FLAG = True
    FIND_ROI = False
    mmpp = 0.075  # millimeters per pixel

    cam_id = "GelSight Mini"
    dev = gsdevice.Camera(gsdevice.Finger.MINI, cam_id)
    dev.connect()
    net_file_path = 'nnmini.pt'
    net_path = os.path.join('.', net_file_path)
    print('net path = ', net_path)

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15, dev)
    net = nn.load_nn(net_path, gpuorcpu)

    if SAVE_VIDEO_FLAG:
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (160, 120), isColor=True)

    f0 = dev.get_raw_image()
    if FIND_ROI:
        roi = cv2.selectROI(f0)
        cv2.imshow('ROI', f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])])
        print('Press q in ROI image to continue')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        roi = (30, 50, 186, 190) if f0.shape == (320, 240, 3) else (0, 0, f0.shape[1], f0.shape[0])

    print('roi = ', roi)
    print('Press q on image to exit')

    if device == 'mini':
        vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp)
    else:
        vis3d = gs3drecon.Visualize3D(dev.imgw, dev.imgh, '', mmpp)

    try:
        while True:
            f1 = dev.get_image(roi)
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)
            boundary = np.array([-4.5, -4])
            bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            bigframe = choose_boundary(dm, bigframe, boundary)
            cv2.imshow('Image', bigframe)
            vis3d.update(dm)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

    except KeyboardInterrupt:
        print('Interrupted!')
        dev.stop_video()

