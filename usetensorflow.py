# /usr/bin/python
# -*- encoding:utf-8 -*-


import tensorflow as tf
from tesorflow.contrib import learn

classifier=learn.DNNClassifier(hidden_units=[10,20,10],n_classes=3)
classifier.fit(x_train,y_train,steps=200)