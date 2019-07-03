from collections import namedtuple

Params = namedtuple("Config", [
  'img_rows',
  'img_cols',
  'network',
  'task',
  'loss',
  'lr',
  'optimizer',
  'batch_size',
  'epoch_size',
  'CLAHE',
  'nb_epoch',
  'cycle_start_epoch',
  'predict_batch_size',
  'CROP',
  'Flip',
  'lighting',
  'affine',
  'randcrop',
  'perspective',
  'dbg',
  'save_images',
  'include_top',
  'weights',
  'data_path',
  'data_path_test'

])