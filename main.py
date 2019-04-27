from importlib import reload
from utils import *
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
from config import *
from tensorflow.contrib.layers import flatten
from PIL import Image
import os
from extract import *


# 加载训练集，验证集，测试集
def load_data():
	# 从文件分别加载
	with open(training_file, mode='rb') as f:
		train = pickle.load(f)
	with open(validation_file, mode='rb') as f:
		valid = pickle.load(f)
	with open(testing_file, mode='rb') as f:
		test = pickle.load(f)
	X_train, y_train = train['features'], train['labels']
	X_valid, y_valid = valid['features'], valid['labels']
	X_test, y_test = test['features'], test['labels']
	# Let's print a few of those mappings now
	sign_names = pd.read_csv(signs_name_path)
	sign_names.set_index("ClassId")
	sign_names.head(n=3)
	X_train_id_to_label = group_img_id_to_lbl(y_train, sign_names)
	# We should group by label id to understand the distribution
	X_train_group_by_label_count = group_img_id_to_lb_count(X_train_id_to_label)
	X_train_group_by_label_count.head(n=5)
	X_train_group_by_label_count.plot(kind='bar', figsize=(15, 7))
	X_train_group_by_label = X_train_id_to_label.groupby(["label_id", "label_name"])
	img_per_class = 5
	
	show_random_dataset_images(X_train_group_by_label, X_train)
	print("Features shape: ", X_train.shape)
	print("Labels shape: ", y_train.shape)
	# 训练集的数量
	n_train = X_train.shape[0]
	# 验证集的数量
	n_validation = X_valid.shape[0]
	# 测试集的数量
	n_test = X_test.shape[0]
	# 训练集的图片尺寸
	image_shape = X_train.shape[1:]
	# How many unique classes/labels there are in the dataset.
	n_classes = len(set(y_train))
	print("Number of training examples =", n_train)
	print("Number of validation examples =", n_validation)
	print("Number of testing examples =", n_test)
	print("Image data shape =", image_shape)
	print("Number of classes =", n_classes)
	X_train_grayscale_normalised = preprocess_features(X_train)
	X_valid_grayscale_normalised = preprocess_features(X_valid)
	X_test_grayscale_normalised = preprocess_features(X_test)
	# 从csv文件读取类别和对应的中文名
	
	return sign_names, n_classes, X_train_grayscale_normalised, y_train, X_valid_grayscale_normalised, y_valid, X_test_grayscale_normalised, y_test


# LeNet模型的配置
class ModelConfig:
	# 进行初始化配置
	def __init__(self, model, name, input_img_dimensions, conv_layers_config, fc_output_dims, output_classes, dropout_keep_pct):
		"""
		:param model: 模型名称，"EdLeNet"
		:param name: 模型文件保存名称，"SEdLeNet_3x3_Dropout_0.50"
		:param input_img_dimensions: 输入图片维度 [41912. 32, 32, 3]
		:param conv_layers_config: 卷积层配置 [3, 32, 3] 分别代表卷积核过滤器尺寸为3x3，初始卷积深度为32，总共由3个卷积层
		:param fc_output_dims: 全连接层维度 [120, 84] 分别表示第一层全连接层神经元节点为120，第二层全连接层神经元节点为84
		:param output_classes: 最终预测的种类数 20
		:param dropout_keep_pct: 随机丢弃率 [60, 50] 分别表示卷积层丢弃率为60%， 全连接层丢弃率为50%
		"""
		self.model = model
		self.name = name
		self.input_img_dimensions = input_img_dimensions
		# Determines the wxh dimension of filters, the starting depth (increases by x2 at every layer)
		# and how many convolutional layers the network has
		self.conv_filter_size = conv_layers_config[0]
		self.conv_depth_start = conv_layers_config[1]
		self.conv_layers_count = conv_layers_config[2]
		
		self.fc_output_dims = fc_output_dims
		self.output_classes = output_classes
		
		# Try with different values for drop out at convolutional and fully connected layers
		self.dropout_conv_keep_pct = dropout_keep_pct[0]
		self.dropout_fc_keep_pct = dropout_keep_pct[1]


# EDLeNet模型执行器
class ModelExecutor:
	def __init__(self, model_config, learning_rate=0.001):
		"""
		:param model_config: 加载EDLeNet模型配置，即ModelConfig
		:param learning_rate: 神经网络学习率
		"""
		self.model_config = model_config
		self.learning_rate = learning_rate
		# 初始化神经网络结构图
		self.graph = tf.Graph()
		with self.graph.as_default() as g:
			with g.name_scope(self.model_config.name) as scope:
				# 通过该函数创建EDLeNet模型操作
				self.create_model_operations()
				# 初始化保存模型操作
				self.saver = tf.train.Saver()
	
	def create_placeholders(self):
		# 神经网络输入矩阵维度， [41912, 32, 32, 1]
		input_dims = self.model_config.input_img_dimensions
		# 定义输入placeholder， [？, 32, 32, 1]
		self.x = tf.placeholder(tf.float32, (None, input_dims[0], input_dims[1], input_dims[2]),
		                        name="{0}_x".format(self.model_config.name))
		# 定义输出placeholder， [?]
		self.y = tf.placeholder(tf.int32, (None), name="{0}_y".format(self.model_config.name))
		# 将预测的全部类别映射到数组
		self.one_hot_y = tf.one_hot(self.y, self.model_config.output_classes)
		# 卷积层丢弃率
		self.dropout_placeholder_conv = tf.placeholder(tf.float32)
		# 全连接层丢弃率
		self.dropout_placeholder_fc = tf.placeholder(tf.float32)
	
	def create_model_operations(self):
		# 创建神经网络初始placeholder
		self.create_placeholders()
		# 将modelconfig中的model，即EDLeNet定义为cnn
		cnn = self.model_config.model
		# cnn(即EDLeNet)的回归模型
		self.logits = cnn(self.x, self.model_config, self.dropout_placeholder_conv, self.dropout_placeholder_fc)
		# cnn(即EDLeNet)的交叉熵
		self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=self.logits)
		# cnn(即EDLeNet)的损失操作
		self.loss_operation = tf.reduce_mean(self.cross_entropy)
		# cnn(即EDLeNet)的优化器定义为Adam
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		# cnn(即EDLeNet)的训练操作
		self.training_operation = self.optimizer.minimize(self.loss_operation)
		# cnn(即EDLeNet)的计算预测准确率操作
		self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
		# cnn(即EDLeNet)的计算准确率操作
		self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		# cnn(即EDLeNet)的最终预测操作
		self.prediction = tf.argmax(self.logits, 1)
		#self.top5_predictions = tf.nn.top_k(tf.nn.softmax(self.logits), k=5, sorted=True, name=None)
	
	# 执行模型函数, 计算模型平均准确率和平均损失值
	def evaluate_model(self, X_data, Y_data, batch_size):
		"""
		:param X_data: 输入到神经网络的数据
		:param Y_data: 输出数据
		:param batch_size: 每次训练加载的数量， 512
		:return: 模型平均准确率，模型平均损失值
		"""
		num_examples = len(X_data)
		total_accuracy = 0.0
		total_loss = 0.0
		sess = tf.get_default_session()
		# 按batch_size大小分批加载训练数据
		for offset in range(0, num_examples, batch_size):
			batch_x, batch_y = X_data[offset:offset + batch_size], Y_data[offset:offset + batch_size]
			# Compute both accuracy and loss for this batch
			# 运行模型计算准确率操作
			accuracy = sess.run(self.accuracy_operation,
			                    feed_dict={self.dropout_placeholder_conv: 1.0,
			                               self.dropout_placeholder_fc: 1.0,
			                               self.x: batch_x,
			                               self.y: batch_y
			                               })
			# 运行模型计算损失率操作
			loss = sess.run(self.loss_operation, feed_dict={
				self.dropout_placeholder_conv: 1.0,
				self.dropout_placeholder_fc: 1.0,
				self.x: batch_x,
				self.y: batch_y
			})
			# 计算模型平均准确率和平均损失值
			total_accuracy += (accuracy * len(batch_x))
			total_loss += (loss * len(batch_x))
		return total_accuracy / num_examples, total_loss / num_examples
	
	# 神经网络训练模型函数
	def train_model(self, X_train_features, X_train_labels, X_valid_features, y_valid_labels, batch_size=512, epochs=100, PRINT_FREQ=10):
		"""
		:param X_train_features: 输入训练集， [?, 32, 32, 1]
		:param X_train_labels: 输出训练集分类标签, [?]
		:param X_valid_features: 输入验证集， [?, 32, 32, 1]
		:param y_valid_labels: 输出验证集分类标签, [?]
		:param batch_size: 分批加载训练数据大小，512
		:param epochs: 模型训练迭代次数
		:param PRINT_FREQ: 打印结果频率，默认十次迭代输出一次结果
		:return: 训练结果矩阵（每次迭代训练集准确率和损失值）， 验证结果矩阵（每次迭代验证集准确率和损失值）， 迭代周期
		"""
		# 初始化训练结果矩阵， 验证结果矩阵
		training_metrics = np.zeros((epochs, 3))
		validation_metrics = np.zeros((epochs, 3))
		# 开启tensorflow会话
		with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
			# 加载tensorflow全局初始化函数
			sess.run(tf.global_variables_initializer())
			num_examples = len(X_train_features)
			print("Training {0} [epochs={1}, batch_size={2}]...\n".format(self.model_config.name, epochs, batch_size))
			# 进入训练迭代过程
			for i in range(epochs):
				start = time.time()
				# 打乱训练集和每张图片对应的标签
				X_train, Y_train = shuffle(X_train_features, X_train_labels)
				# 分批加载训练数据
				for offset in range(0, num_examples, batch_size):
					end = offset + batch_size
					batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
					# 运行训练模型操作
					sess.run(self.training_operation, feed_dict={
						self.x: batch_x,
						self.y: batch_y,
						self.dropout_placeholder_conv: self.model_config.dropout_conv_keep_pct,
						self.dropout_placeholder_fc: self.model_config.dropout_fc_keep_pct,
					})
				end_training_time = time.time()
				# 计算训练周期
				training_duration = end_training_time - start
				# 调用执行模型函数计算训练集准确率和训练损失值
				training_accuracy, training_loss = self.evaluate_model(X_train_features, X_train_labels, batch_size)
				# 调用执行模型函数计算验证集准确率和训练损失值
				validation_accuracy, validation_loss = self.evaluate_model(X_valid_features, y_valid_labels, batch_size)
				end_epoch_time = time.time()
				# 计算验证周期
				validation_duration = end_epoch_time - end_training_time
				# 迭代周期
				epoch_duration = end_epoch_time - start
				# 每十次迭代打印一次结果
				if i == 0 or (i + 1) % PRINT_FREQ == 0:
					print("[{0}]\ttotal={1:.3f}s | train: time={2:.3f}s, loss={3:.4f}, acc={4:.4f} | val: time={5:.3f}s, loss={6:.4f}, acc={7:.4f}".format(
							i + 1, epoch_duration, training_duration, training_loss, training_accuracy,
							validation_duration, validation_loss, validation_accuracy))
				
				training_metrics[i] = [training_duration, training_loss, training_accuracy]
				validation_metrics[i] = [validation_duration, validation_loss, validation_accuracy]
			
			model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
			# 保存模型
			self.saver.save(sess, model_file_name)
			print("Model {0} saved".format(model_file_name))
		return training_metrics, validation_metrics, epoch_duration
	
	# 测试模型函数，计算测试集的准确率和损失值以及运行周期
	def test_model(self, test_imgs, test_lbs, batch_size=512):
		"""
		:param test_imgs: 测试集数据
		:param test_lbs: 测试集标签
		:param batch_size: 分批加载数据大小，512
		:return: 测试集的准确率， 损失值， 运行周期
		"""
		# 开启tensorflow会话
		with tf.Session(graph=self.graph) as sess:
			# 初始化tensorflow，并恢复之前保存的模型
			tf.global_variables_initializer()
			model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
			self.saver.restore(sess, model_file_name)
			start = time.time()
			# 调用执行模型函数，计算测试集的准确率和损失值
			(test_accuracy, test_loss) = self.evaluate_model(test_imgs, test_lbs, batch_size)
			duration = time.time() - start
			print("[{0} - Test Set]\ttime={1:.3f}s, loss={2:.4f}, acc={3:.4f}".format(self.model_config.name, duration, test_loss, test_accuracy))
		return test_accuracy, test_loss, duration
	
	# 预测结果函数
	def predict(self, imgs):
		"""
		:param imgs: 预测的图片
		:return: 每张图片对应的标签， []
		"""
		preds = []
		# 开启tensorflow会话
		with tf.Session(graph=self.graph) as sess:
			# 初始化tensorflow，并恢复之前保存的模型
			tf.global_variables_initializer()
			model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
			self.saver.restore(sess, model_file_name)
			# 执行模型预测操作，预测最终结果
			preds = sess.run(self.prediction, feed_dict={
				self.x: imgs,
				self.dropout_placeholder_conv: 1.0,
				self.dropout_placeholder_fc: 1.0
			})
		return preds

	
# 定义EDLeNet网络结构
def EdLeNet(x, mc, dropout_conv_pct, dropout_fc_pct):
	"""
	:param x:  先前定义的模型的输入placeholder, [?, 32, 32, 1]
	:param mc: 模型配置，modelconfig
	:param dropout_conv_pct: 卷积层丢弃率， 60%
	:param dropout_fc_pct: 全连接层丢弃率， 50%
	:return: 神经网络回归模型
	"""
	mu = 0
	sigma = 0.1
	prev_conv_layer = x
	# 第一层卷积层的卷积核深度， 32
	conv_depth = mc.conv_depth_start
	# 卷积层的层数， 3
	conv_input_depth = mc.input_img_dimensions[-1]
	print("[EdLeNet] Building neural network [conv layers={0}, conv filter size={1}, conv start depth={2}, fc layers={3}]".format(
			mc.conv_layers_count, mc.conv_filter_size, conv_depth, len(mc.fc_output_dims)))
	# 循环三次，创建三层卷积层
	for i in range(0, mc.conv_layers_count):
		# 每一层卷积层的卷积核的输出深度为上一层的两倍
		conv_output_depth = conv_depth * (2 ** i)
		# 定义每一层卷积层的权重
		conv_W = tf.Variable(tf.truncated_normal(shape=(mc.conv_filter_size, mc.conv_filter_size, conv_input_depth, conv_output_depth),
			                    mean=mu, stddev=sigma))
		# 定义每一层卷积层的偏置
		conv_b = tf.Variable(tf.zeros(conv_output_depth))
		# 定义每一层的卷积操作后的特征图
		conv_output = tf.nn.conv2d(prev_conv_layer, conv_W, strides=[1, 1, 1, 1], padding='VALID', name="conv_{0}".format(i)) + conv_b
		# 定义每一卷积层层的激励函数
		conv_output = tf.nn.relu(conv_output, name="conv_{0}_relu".format(i))
		# 定义每一层的卷积层的池化操作
		conv_output = tf.nn.max_pool(conv_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		# 定义每一卷积层的丢弃操作
		conv_output = tf.nn.dropout(conv_output, dropout_conv_pct)
		# 下一层卷积层的输入等于该层的输出
		prev_conv_layer = conv_output
		# 下一层卷积层的输入卷积核深度等于该层的输出卷积核深度
		conv_input_depth = conv_output_depth
	# 将卷积层的最后一层拉成一维矩阵
	fc0 = flatten(prev_conv_layer)
	# 定义全连接层的开始为fc0
	prev_layer = fc0
	# 循环两个全连接层，[120, 84]
	for output_dim in mc.fc_output_dims:
		# 定义全连接层的权重
		fcn_W = tf.Variable(tf.truncated_normal(shape=(prev_layer.get_shape().as_list()[-1], output_dim), mean=mu, stddev=sigma))
		# 定义全连接层的偏置
		fcn_b = tf.Variable(tf.zeros(output_dim))
		# 定义全连接层的丢弃操作
		prev_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(prev_layer, fcn_W) + fcn_b), dropout_fc_pct)
	# 最后一层输出层的权重
	fc_final_W = tf.Variable(tf.truncated_normal(shape=(prev_layer.get_shape().as_list()[-1], mc.output_classes), mean=mu, stddev=sigma))
	# 最后一层输出层的偏置
	fc_final_b = tf.Variable(tf.zeros(mc.output_classes))
	# 计算整个模型架构的回归模型
	logits = tf.matmul(prev_layer, fc_final_W) + fc_final_b
	return logits


# 从文件夹中读取待测试文件并提取途中交通标志
def get_imgs_from_folder(path, grayscale=False):
	"""
	:param path: 预测图片的文件夹路径
	:param grayscale: 是否进行灰度化
	:return: 一张图片中有几个交通标志，图片中交通标志的位置，处理后的交通标志部分
	"""
	# 加载文件夹中的jpg和png类型的所有图片
	img_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".png")]
	_images = []
	_images_sign_num = {}
	rects = []
	# 循环对每张图片进行操作
	for i, img_path in enumerate(img_list):
		# 调用提取交通标志函数， _rect为图片中交通标志的位置，_img为截取到的交通标志部分
		_rects, _img = extract_image(img_path)
		_images.extend(_img)
		_images_sign_num[img_path] = len(_rects)
		rects.extend(_rects)
		if _images is None:
			print("No sign be detected!!!")
	# 对提取到的交通标志部分进行灰度处理
	if grayscale:
		imgs = np.asarray(list(map(lambda img: to_grayscale(img), _images)))
	else:
		imgs = np.array(_images)
	# 对提取到的交通标志部分进行归一化和直方图均衡化
	ret = preprocess_features(imgs)
	return _images_sign_num, rects, ret

	
# 在图片中框出交通标志和预测结果
def draw_rects(_images_sign_num, rects, sign_label):
	_tmp = 0
	_label_index = 0
	img_index = 0
	for img_path, _num in _images_sign_num.items():
		img = cv.imread(img_path)
		if img is None:
			continue
		img_b = img.copy()
		for rect in rects[_tmp:_tmp + _num]:
			cv.rectangle(img_b, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
			cv.putText(img_b, sign_label[_label_index] + " ", (rect[0], rect[1]), 1, 1.5, (0, 0, 255), 2)
			cv.imwrite('custom_images/results' + "/detect_result%d.jpg" % img_index, img_b)
			_label_index += 1
		_tmp += _num
		img_index += 1


# 主函数
if __name__ == "__main__":
	# # 调用从文件夹中提取预测的交通标志函数，_images_sign_num为一张图片中有几个交通标志，rects为图片中交通标志的位置，new_imgs为处理后的交通标志部分
	# _images_sign_num, rects, new_imgs = get_imgs_from_folder(new_imgs_dir)
	# # 调用加载训练集，验证集，测试集的函数，并分别返回标签名称和预测总类别，以及数据预处理过后的结果
	sign_names, n_classes, X_train_grayscale_normalised, y_train, X_valid_grayscale_normalised, \
	y_valid, X_test_grayscale_normalised, y_test = load_data()
	
	# model_name = "SEdLeNet_3x3_Dropout_0.50"
	# # 调用ModelConfig类初始化模型配置
	# model_config = ModelConfig(EdLeNet, model_name, [32, 32, 1], [3, 32, 3], [120, 84], n_classes, [0.6, 0.5])
	# # 调用ModelExecutor类初始化模型执行器
	# model_executor = ModelExecutor(model_config)
	#
	# model_path = models_path + "%s.chkpt.meta" % model_name
	# # 判断是否存在已保存的训练模型
	# if not os.path.exists(model_path):
	# 	# 此处表示不存在已保存的训练模型，则调用模型执行器中的训练模型函数，返回值分别为训练结果矩阵（每次迭代训练集准确率和损失值）， 验证结果矩阵（每次迭代验证集准确率和损失值）， 迭代周期
	# 	tr_metrics, val_metrics, duration = model_executor.train_model(X_train_grayscale_normalised, y_train
	# 	                                                               , X_valid_grayscale_normalised, y_valid,
	# 	                                                               epochs=100)
	# 	# 调用模型执行器中的测试集预测函数，返回值分别为测试集的准确率， 损失值， 运行周期
	# 	ts_accuracy, ts_loss, ts_duration = model_executor.test_model(X_test_grayscale_normalised, y_test)
	# 	metrics_arr = [tr_metrics, val_metrics]
	# 	lbs = ["3x3 training", "3x3 validation "]
	# 	# 调用画图函数，画出训练的准确率和损失值变化图
	# 	plot_model_results(metrics_arr, [2, 1], lbs, ["Epochs", "Epochs"], ["Accuracy", "Loss"],
	# 	                   ["Accuracy vs Epochs", "Loss vs Epochs"],
	# 	                   "Grayscale Histogram-Equalized - Accuracy and Loss of models", fig_size=(17, 5))
	# else:
	# 	# 此处为存在已保存的训练模型，则导入保存的模型
	# 	saver = tf.train.import_meta_graph(model_path)
	# 	with tf.Session(graph=model_executor.graph) as sess:
	# 		model_executor.saver.restore(sess, models_path + "%s.chkpt" % model_name)
	# # 调用模型执行器中的预测函数，对先前从文件夹中检测到的交通标志传入该函数，返回预测到的结果
	# preds = model_executor.predict(new_imgs)
	# print(preds)
	# # 将返回的标签值对应到中文名称
	# sign_label = list(map(lambda cid: sign_names[sign_names["ClassId"] == cid]["SignName"].values[0],  preds))
	# print(sign_label)
	# # 将检测到的交通标志和预测的类别画在图中
	# draw_rects(_images_sign_num, rects, sign_label)