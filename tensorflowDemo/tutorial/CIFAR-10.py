import cifar10
from tensorflowDemo import utils

from tensorflowDemo.utils import *
#cifar10.maybe_download_and_extract()

class_names=utils.load_class_names(filename='batches.meta')
print(class_names)


images_train, cls_train, labels_train = utils.load_training_data()

images_test, cls_test, labels_test = utils.load_test_data()

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

img_size_cropped=24


