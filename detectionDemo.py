import convolutionalNN as cnn
import dataset
import tensorflow as tf
import os
import cv2
import numpy as np
import sys

##### SAVING A GRAPH #####

"""#Prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1= tf.Variable(2.0,name="bias")
feed_dict ={w1:4,w2:8}

#Define a test operation that we will restore
w3 = tf.add(w1,w2)
w4 = tf.multiply(w3,b1,name="op_to_restore")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Create a saver object which will save all the variables
saver = tf.train.Saver()

#Run the operation by feeding input
print(sess.run(w4,feed_dict))
#Prints 24 which is sum of (w1+w2)*b1

#Now, save the graph
saver.save(sess, 'my_test_model',global_step=1000)

##### RESTORING A GRAPH #####

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))


# Access saved Variables directly
print(sess.run('bias:0'))
# This will print 2, which is the value of bias that we saved


# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict ={w1:13.0,w2:17.0}

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

print(sess.run(op_to_restore,feed_dict))
#This will print 60 which is calculated """

# First, pass the path of the image
image_size_x=128
image_size_y=96
num_channels=3
#images = []
cap = cv2.VideoCapture(1)
## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('results/steering_model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('results/'))
# Accessing the default graph which we have restored
graph = tf.get_default_graph()


while(True):
    # Reading the image using OpenCV
    _, frame = cap.read()
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(frame, (image_size_x, image_size_y), 0, 0, cv2.INTER_LINEAR)
    #images.append(image)
    #images = np.array(images, dtype=np.uint8)
    #images = images.astype('float32')
    #images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = image.reshape(1, image_size_y, image_size_x, num_channels)

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, len(os.listdir('dataSet/imgs/'))))


    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    print(result)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bwframe = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)  # SOGLIA CIRCA 220
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bwframeOrig = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    resultArray = np.argmax(result)

    cv2.putText(bwframeOrig,"Prediction: " + str(resultArray), (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 230)

    # Display the resulting frame
    cv2.imshow('frame', bwframeOrig)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()