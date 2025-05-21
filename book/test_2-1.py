from keras.datasets import mnist  
import code

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

code.interact(local=locals())