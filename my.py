from hdr_plus import process_stack, hdrplus_tiled
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import argparse
from isp import RAW2RGB
import cv2

parser = argparse.ArgumentParser(description='')
parser.add_argument('--burst_dir', dest='burst_dir', default='./data/', help='filename')
parser.add_argument('--postfix', dest='postfix', default='RAWMIPI', help='')
parser.add_argument('--width', dest='width', default=4000, type=int, help='')
parser.add_argument('--height', dest='height', default=3000, type=int, help='')
parser.add_argument('--c', dest='c', default=4e8, type=float, help='')
args = parser.parse_args()

def mipirawtorawf(raw, h):
    ###h for height of the image
    raw10 = raw.reshape(h, -1, 5).astype(np.uint16)
    a,b,c,d,e = [raw10[...,x] for x in range(5)]
    x1 = (a << 2) + ((e >> 0) & 0x03)
    x2 = (b << 2) + ((e >> 2) & 0x03)
    x3 = (c << 2) + ((e >> 4) & 0x03)
    x4 = (d << 2) + ((e >> 6) & 0x03)

    x1 = x1.reshape(h, -1, 1)
    x2 = x2.reshape(h, -1, 1)
    x3 = x3.reshape(h, -1, 1)
    x4 = x4.reshape(h, -1, 1)

    x = np.dstack((x1, x2, x3, x4))
    x = x.reshape(h, -1)
    return x / np.float(2**10-1)

burst = sorted(glob('{}*.{}'.format(args.burst_dir, args.postfix)))
images = list()
frame_num = len(burst)
for i in range(frame_num):
    if args.postfix == 'raw':
        buffer = open(burst[i], 'rb').read()
        buffer_np = np.frombuffer(buffer, dtype=np.uint16)
        Iref = buffer_np.astype(np.float32) / (2 ** 10 - 1)
        Iref = Iref.reshape((args.width, args.height))
        Iref = np.expand_dims(Iref, axis=0)
        images.append(Iref)
    elif args.postfix == 'RAWMIPI':
        buffer = open(burst[i], 'rb').read()
        buffer_np = np.frombuffer(buffer, dtype=np.uint8)
        Iref = mipirawtorawf(buffer_np, args.height)
        Iref = np.expand_dims(Iref, axis=0)
        images.append(Iref)
bls = np.zeros((len(burst), 1))

images = np.concatenate(images, axis=0)
burst_shape = images.shape
tiles, dumb = process_stack(raw_im_list=images, bls=bls, raw_true=None, bl_true=None, bayer=True)

# tiles = np.transpose(tiles, [3, 1, 2, 0])
# c_central, pvals = hdrplus_csearch(noisy=tiles, truth=images[0], N=16, sig=sig_read_single_std, post_fn=None)

tiles = tiles.transpose([3,1,2,0])# [b, h, w, c]
dumb = dumb.transpose([3,1,2,0])
b, h, w, _ = tiles.shape
tiles_packed = np.zeros((b, h * 2, w * 2))
dumb_packed = np.zeros((b, h * 2, w * 2))
tiles_packed[:, ::2, ::2] = tiles[:, :, :, 0]
tiles_packed[:, 1::2, ::2] = tiles[:, :, :, 1]
tiles_packed[:, ::2, 1::2] = tiles[:, :, :, 2]
tiles_packed[:, 1::2, 1::2] = tiles[:, :, :, 3]

dumb_packed[:, ::2, ::2] = dumb[:, :, :, 0]
dumb_packed[:, 1::2, ::2] = dumb[:, :, :, 1]
dumb_packed[:, ::2, 1::2] = dumb[:, :, :, 2]
dumb_packed[:, 1::2, 1::2] = dumb[:, :, :, 3]

sr = 0.00000111187126
ss = 0.0004053641569

tiles_packed = tiles_packed.transpose([1,2,0])# [h, w, b]
sig_read_single_std = sr + np.maximum(0., tiles_packed) * ss
h, w, ch = tiles_packed.shape



# merged_patch = hdrplus_tiled(tiles_packed, h, w, sig_read_single_std, c=args.c)

merged_patch_r = hdrplus_tiled(tiles_packed[::2, ::2, :], h // 2, w // 2, sig_read_single_std, c=args.c)
merged_patch_g1 = hdrplus_tiled(tiles_packed[1::2, ::2, :], h // 2, w // 2, sig_read_single_std, c=args.c)
merged_patch_g2 = hdrplus_tiled(tiles_packed[::2, 1::2, :], h // 2, w // 2, sig_read_single_std, c=args.c)
merged_patch_b = hdrplus_tiled(tiles_packed[1::2, 1::2, :], h // 2, w // 2, sig_read_single_std, c=args.c)

print(merged_patch_r.shape)
plt.figure("in and out")
plt.subplot(121)
plt.imshow(tiles_packed[..., 0])
plt.subplot(122)
plt.imshow(merged_patch_r)
out_dumb = RAW2RGB(merged_patch)
# out_dumb = cv2.cvtColor(out_dumb, cv2.COLOR_RGB2BGR)
# cv2.imwrite("./merged_{}.png".format(args.c), out_dumb.astype(np.uint8))




# plt.figure("sig_read_single_std")
# plt.imshow(sig_read_single_std[...,0])
# plt.figure("tiles_packed")
# plt.imshow(tiles_packed[...,0])









