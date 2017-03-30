#--*# -*- coding: utf-8 -*-

'''
생성 방법
cnn = CNN(n_input,n_classes,dropout,learning_rate)
cnn.setmodel(c,c,c,af)
cnn.addTraningSet(imgfile,n_label) # ("./john/picname.jpg",(0~n_classes))
cnn.learning()
cnn.whatisthis(cv2_img)
'''

import tensorflow as tf
import cv2

class CNN:
    def __init__(self,n_input,n_classes,dropout,learning_rate):

        self.maxnum = 0
        self.n_input = n_input
        self.n_classes = n_classes
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_classes])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([20 * 20 * 64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, n_classes]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        self.pred = self.conv_net(self.x, self.weights, self.biases, self.keep_prob)
        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # Initializing the variables
        self.init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(self.init)

        self.batch = [[],[]]
        self.saver = tf.train.Saver()

    # Create some wrappers for simplicity
    def conv2d(self,x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self,x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # Create model
    def conv_net(self,x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 80, 80, 1])
        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)
        # Output, class prediction
        print fc1
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        print out
        return out

    def addTraningSet(self,imgfile,label):
        self.batch
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pic = []
        for row in img:
            for picxel in row:
                f_gray = float(picxel)/255.0
                pic.append(f_gray)

        self.batch[0].append(pic)
        label_format = [ 1 if x == label else 0 for x in range(self.n_classes)]
        self.batch[1].append(label_format)


    def learning(self):
        step = 0
        loss = 123
        while True:
            batch_x = self.batch[0]
            batch_y = self.batch[1]

            # Run optimization op (backprop)
            self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.dropout})
            if step % 100:
                # Calculate batch loss and accuracy
                loss, acc = self.sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.})
                print("Iter " + str(step) + ", Minibatch Loss= "+"{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

            if loss < 0.1:
                break
            step += 1
        print("Optimization Finished!")

    def whatisthis(self,cv2_img,type = "BGR"):

        #img = cv2.imread("./2828leaningdata/1/1.jpg")
        if type == "gray":
            img = cv2_img
        elif type == "BGR":
            img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

        pic = []
        for row in img:
            for picxel in row:
                f_gray = float(picxel)/255.0
                pic.append(f_gray)
        pic = [pic]
	#if you want to show softmax, write "tf.nn.softmax(self.pred)" or just write "self.pred"
        result = self.sess.run(self.pred, feed_dict={self.x: pic, self.keep_prob: 1.})
        result_list = (list(result)[0])
        mlist = list(result_list)
        temp = mlist[int(mlist.index(max(mlist)))]
        if temp > self.maxnum:
            self.maxnum = temp
        print mlist ,int(mlist.index(max(mlist))) ,self.maxnum
        return int(mlist.index(max(mlist)))

    def save(self):
        save_path = self.saver.save(self.sess, "./model.mcnn")
        print ("Model saved in file: ", save_path)

    def restore(self):
        self.saver.restore(self.sess,"./model.mcnn")

if __name__ =="__main__":
    cnn = CNN(6400, 2, 0.75, 0.001)
    cnn.restore()


    face_cascade = cv2.CascadeClassifier('./haar_face_data.xml')
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    while True:
        ret, img = cap.read()
        # smallImg = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        smallImg = img
        gray = cv2.cvtColor(smallImg, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) != 0:
            for (x, y, w, h) in faces:
                crop_img = gray[y:y + h, x:x + w]
                crop_img = cv2.resize(crop_img,(80,80))
                test_img = crop_img
                temp= cnn.whatisthis(test_img, "gray")
                if  temp == 0:
                    cv2.rectangle(smallImg, (x, y), (x + w, y + h), (255, 0, 0), 2)
                elif temp == 1:
                    cv2.rectangle(smallImg, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("detect sim ", smallImg)
        cv2.waitKey(1)
    cap.release()
