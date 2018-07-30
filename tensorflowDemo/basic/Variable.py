import tensorflow as tf
#a=tf.constant([1.0,2.0],name="a")
a=tf.constant([1,2],name="a",dtype=tf.float32)
b=tf.constant([2.0,3.0],name="b")
result=tf.add(a,b,name="add")
print(result)
#变量一定要指定数据类型,不然会发生数据类型不匹配的问题。