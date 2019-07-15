from hdr_plus import process_stack
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# HDR Plus
def unfold_channels(x, n, axis=-2):

  sh0 = x.shape
  x = np.reshape(x, [1, sh0[axis]//n, n, sh0[axis+1]//n, n, -1])
  x = np.transpose(x, [0, 1, 3, 2, 4, 5])
  sh = x.shape
  x = np.reshape(x, [sh[0], sh[1], sh[2], n*n, -1])
  sh = list(sh0[:axis]) + [sh0[axis]//n, sh0[axis+1]//n, n*n] + \
          (list(sh0[axis+2:]) if axis != -2 else [])
  x = np.reshape(x, sh)

  return x


def psnr(estimate, truth, bd=0, batched=False):
  sqdiff = np.square(estimate - truth)
  if not batched:
    sqdiff = sqdiff[np.newaxis,...]
  if bd != 0:
    sqdiff = sqdiff[:, bd:-bd, bd:-bd, ...]
  sqdiff = np.reshape(sqdiff, [sqdiff.shape[0], -1])
  eps = 1e-10
  val = -10. * (np.log10(np.maximum(1e-10, np.mean(sqdiff, axis=1))))
  return val if batched else np.mean(val)

def down2(im, batched=False):
  if not batched:
    im = im[np.newaxis, ...]
  im = im[:, :im.shape[1]//2*2, :im.shape[2]//2*2, ...]
  im = .25 * (im[:, ::2,::2,...]+
              im[:, 1::2,::2,...]+
              im[:, ::2,1::2,...]+
              im[:, 1::2,1::2,...])
  if not batched:
    im = im[0,...]
  return im

def downN(im, n, batched=False):
  for i in range(n):
    im = down2(im, batched)
  return im


def rcwindow(h, w):
  x = np.linspace(0., w, w, endpoint=False)
  rcw_x = .5 - .5 * np.cos(2. * np.pi * (x + .5) / w)
  y = np.linspace(0., h, h, endpoint=False )
  rcw_y = .5 - .5 * np.cos(2. * np.pi * (y + .5) / h)
  rcw =  rcw_y.reshape((h, 1)) * rcw_x.reshape((1, w))
  return rcw

def hdrplus_merge(imgs, c, sig, spatial=False):
  rcw = rcwindow(imgs.shape[-3], imgs.shape[-2])[...,np.newaxis] #
  plt.figure('rcw')
  plt.subplot(111)
  plt.imshow(rcw[..., 0])
  imgs = imgs * rcw #为什么要先乘一个二维余弦函数

  imgs_f = np.fft.fft2(imgs, axes=(-3,-2))
  Dz2 = np.square(np.abs(imgs_f[...,0:1] - imgs_f))

  Az = Dz2 / (Dz2 + c*sig**2)
  filts = 1 - Az
  filts[...,0] = 1 + np.sum(Az[...,1:], axis=-1)
  output_f = np.mean(imgs_f * filts, axis=-1)
  output_f = np.real(np.fft.ifft2(output_f))

  if spatial:

    output_stack = []
    filts_s = np.real(np.fft.ifft2(filts, axes=(-3,-2)))
    N = imgs.shape[-1]
    for i in range(N):
        in1 = imgs[...,i]
        in2 = filts_s[...,i]
        output_stack.append(np.fft.fftshift(sp.signal.convolve2d(in1, in2, mode='same', boundary='wrap')))
    output_stack = np.stack(output_stack, axis=-1)
    output_stack = np.roll(np.roll(output_stack,-1,axis=0),-1,axis=1)
    output_s = np.mean(output_stack, axis=-1)
    return imgs, output_f, output_s, filts, filts_s, Az, output_stack

  else:
    return imgs, output_f, filts, Az

#这个是[h, w, b]
def hdrplus_tiled(noisy, h, w, sig, c=10**2.5):
  hw = noisy.shape[0:2] #w, h
  buffer = np.zeros_like(noisy[...,0])
  for i in range(2):
    for j in range(2):
      nrolled = np.roll(np.roll(noisy, axis=0, shift=-h // 2 * i), axis=1, shift=-w // 2 * j)
      hpatches = (np.transpose(np.reshape(nrolled, [hw[0]//h, h, hw[1]//w, w, -1]), [0,2,1,3,4]))
      merged = hdrplus_merge(hpatches, c, sig, spatial=False)[1]
      merged = (np.reshape(np.transpose(merged, [0,2,1,3]), hw))
      merged = np.roll(np.roll(merged, axis=0, shift=h//2*i), axis=1, shift=w//2*j)
      buffer += merged
  return buffer


def hdrplus_tiled_sigbatch(noisy, N, sig, c=10**2.5):
  sh = noisy.shape[0:3]

  chunk = 16
  if sh[0] > chunk:
    buffer = []
    print ('tiling the', sh[0], 'last axis')
    for i in range(0, sh[0], chunk):
      print (i),
      buffer.append(hdrplus_tiled_sigbatch(noisy[i:i+chunk, ...], N, sig[i:i+chunk,...], c))
    print ('done')
    return np.concatenate(buffer, axis=0)

  buffer = np.zeros_like(noisy[...,0])
  noisy_ = noisy
  noisy = np.concatenate([noisy, sig], axis=-1)
  for i in range(2):
    for j in range(2):
      nrolled = np.roll(np.roll(noisy, axis=1, shift=-N//2*i), axis=2, shift=-N//2*j)
      hpatches = (np.transpose(np.reshape(nrolled, [sh[0], sh[1]//N, N, sh[2]//N, N, -1]), [0,1,3,2,4,5]))

      patches = hpatches[...,:-1]
      sigpatches = hpatches[...,-1]
      sigpatches = np.sqrt(np.mean(np.mean(sigpatches**2, axis=-1), axis=-1))
      sigpatches = np.tile(np.reshape(sigpatches, list(sigpatches.shape) + [1,1,1]), [1,1,1,N,N,noisy_.shape[-1]])

      merged = hdrplus_merge(patches, c, sigpatches, spatial=False)[1]
      merged = (np.reshape(np.transpose(merged, [0,1,3,2,4]), sh))
      merged = np.roll(np.roll(merged, axis=1, shift=N//2*i), axis=2, shift=N//2*j)
      buffer += merged
  return buffer


def hdrplus_csearch(noisy, truth, N, sig, post_fn=None):
  c_central = 0.
  c_ranges = [np.linspace(-10,10,25), np.linspace(-1,1,25)]
  pvals = []
  for i, c_range in enumerate(c_ranges):
    recons = [hdrplus_tiled(noisy, N, sig, c=10**c) for c in c_central + c_range]
    if post_fn is not None:
      recons = map(post_fn, recons)
    psnrs = [psnr(r, truth) for r in recons]
    pvals.append([c_central + c_range, psnrs])
    c_central = c_central + c_range[np.argmax(psnrs)]
  return c_central, pvals

# Alignment
def small_align(img0, img1, y, x, ys, xs, bd, dy=0, dx=0):
  vals = []
  indices = []
  tile0 = img0[y:y+ys, x:x+xs]
  for i in range(y-bd, y+bd+1):
    for j in range(x-bd, x+bd+1):
      tile1 = img1[i+dy:i+dy+ys, j+dx:j+dx+xs]
      vals.append(np.mean((tile0-tile1)**2))
      indices.append([i,j])
  vals = np.array(vals)
  ind = np.argmin(vals)
  ind = indices[ind]
  vals = vals.reshape([2*bd+1,2*bd+1])
  tile1 = img1[ind[0]:ind[0]+ys, ind[1]:ind[1]+xs]
  return tile1, ind, vals, tile0


def roll2(x, i, j):
  return np.roll(np.roll(x, i, 0), j, 1)


def whole_img_align(img0, img1, bd, pd, verbose=False):
  img0 = np.mean(img0.reshape(list(img0.shape[:2])+[-1]), axis=-1)
  img1 = np.mean(img1.reshape(list(img1.shape[:2])+[-1]), axis=-1)
  vals = []
  indices = []
  tile0 = img0[pd:-pd, pd:-pd, ...]
  for i in range(-bd, bd+1):
    for j in range(-bd, bd+1):
      tile1 = roll2(img1, -i, -j)
      tile1 = tile1[pd:-pd, pd:-pd, ...]
      diff2 = (np.square(tile0-tile1))
      diff = np.mean(diff2)
      vals.append(diff)
      indices.append([bd+i,bd+j])
  vals = np.array(vals)
  ind = np.argmin(vals)
  ind = np.array(indices[ind]) - bd
  vals = vals.reshape([2*bd+1,2*bd+1])
  return ind, vals

def coarse2fine_align(img0, img1, bd, N=2):
  bd0 = bd // 2**N
  img1_ = img1 + 0.
  ind0, vals = whole_img_align(downN(img0, N), downN(img1, N), bd0, bd0)
  ind0 = np.array(ind0) * 2**N
  img1 = roll2(img1, -ind0[0], -ind0[1])
  ind, vals = whole_img_align(img0, img1, 2**N, bd)
  img1 = np.roll(np.roll(img1, -ind[0], 0), -ind[1], 1)
  pd = bd
  tile0 = img0[pd:-pd, pd:-pd, ...]
  tile1 = img1[pd:-pd, pd:-pd, ...]
  tile1_ = img1_[pd:-pd, pd:-pd, ...]
  print ('coarse {}, fine {}, net {}. gain of {} vs {}'.format(
      ind0, ind, ind0+ind,
      np.mean(np.square(tile1-tile0)),
      np.mean(np.square(tile1_-tile0))))
  return tile1, tile0

def mask_hot_pixels(im, thresh=.98):
  k = thresh
  mask = (im >= roll2(im,0,1)*k) * (im >= roll2(im,1,0)*k) * (im >= roll2(im,0,-1)*k)* (im >= roll2(im,-1,0)*k)
  mask = np.prod(mask, axis=-1)
  mask = np.tile(mask[...,np.newaxis], [1,1,im.shape[-1]])
  rep = .25*(roll2(im,0,1)+roll2(im,1,0)+roll2(im,0,-1)+roll2(im,-1,0))
  im[mask==1] = rep[mask==1]
  return im, mask[...,0]


def process_stack(raw_im_list, bls, raw_true=None, bl_true=None, bayer=False):
  # Average bayer blocks
  # Subtract black level
  if bayer:
    imsdn = np.stack([unfold_channels(im, 2, axis=0) for im in raw_im_list], axis=-1)
  else:
    imsdn = np.stack([im for im in raw_im_list], axis=-1)
  print ('Working with size {}'.format(imsdn.shape))
  print ('Average black level {}'.format(np.mean(bls)))
  bls = bls.reshape([1,1,bls.shape[0],bls.shape[-1]]).transpose([0,1,3,2])
  if not bayer:
    bls = np.mean(bls, -2)
  imsdn = imsdn - bls

  # Repress hot pixels
  # mask = np.ones_like(imsdn[...,0])
  # k = .98 # threshold for hotness
  # for i in range(8):
  #   im = imsdn[...,i].astype(np.float64)
  #   m = (im >= roll2(im,0,1)*k) * (im >= roll2(im,1,0)*k) * (im >= roll2(im,0,-1)*k)* (im >= roll2(im,-1,0)*k)
  #   mask = mask * m

  # for i in range(8):
  #   im = imsdn[...,i]
  #   rep = .25*(roll2(im,0,1)+roll2(im,1,0)+roll2(im,0,-1)+roll2(im,-1,0))
  #   im[mask==1] = rep[mask==1]
  #   imsdn[...,i] = im

  imsdn, mask = mask_hot_pixels(imsdn)
  print ('Percent hot pixels = {:.4f}%'.format(100.*np.sum(mask==1)/mask.size))

  # Whole image align
  tiles = []
  bd = 16
  dumb = []
  for i in range(0,5):
    dumb.append(imsdn[bd:-bd,bd:-bd,...,i])
    tile1, tile0 = coarse2fine_align(imsdn[...,0], imsdn[...,i], bd=bd)
    tiles.append(tile1 if i > 0 else tile0)

### FIX below
  tiles = np.stack(tiles, axis=2)
  dumb = np.stack(dumb,axis=2)
  if bayer:
    tiles = tiles.transpose([3,0,1,2])
    dumb = dumb.transpose([3,0,1,2])
  tiles_score = np.mean(np.square(tiles - tiles[...,0:1]))
  dumb_score = np.mean(np.square(dumb - dumb[...,0:1]))

  print ('Alignment complete, total gain of {} vs {}'.format(tiles_score, dumb_score))

  if raw_true is None:
    return tiles, dumb
