import tarfile
import os
import tensorflow as tf
from keras.utils import img_to_array

# Creates and loads pretrained deeplab model
tarball_path = "models/deeplab_model.tar.gz"
graph_def = None
graph = tf.compat.v1.Graph()

# Extract frozen graph from tar archive
tar_file = tarfile.open(tarball_path)
for tar_info in tar_file.getmembers():
    if 'frozen_inference_graph' in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break
tar_file.close()

# Check if the inference graph was found
if graph_def is None:
    raise RuntimeError('Cannot find inference graph in tar archive.')

# Create tensorflow session from the imported graph
with graph.as_default():
    tf.import_graph_def(graph_def, name='')
DEEPLAB_SESSION = tf.compat.v1.Session(graph=graph)

def preprocess_image(image, h=512, w=512):
    # resize with padding so it's not deform the image
    image = tf.image.resize_with_pad(image, target_height=h, target_width=w)

    return img_to_array(image)

def run_deeplab(image, h=512, w=512):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    # Preprocess the image
    image_array = preprocess_image(image, h, w)
    # Run inference
    batch_seg_map = DEEPLAB_SESSION.run(OUTPUT_TENSOR_NAME, feed_dict={INPUT_TENSOR_NAME: [image_array]})
    seg_map = batch_seg_map[0]
    return seg_map