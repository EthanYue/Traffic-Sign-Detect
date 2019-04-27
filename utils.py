import pickle
from skimage import transform
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os


# 画图函数，画出训练的准确率和损失值变化图
def plot_model_results(metrics, axes, lbs, xlb, ylb, titles, fig_title, fig_size=(7, 5), epochs_interval=10):
	fig, axs = plt.subplots(nrows=1, ncols=len(axes), figsize=fig_size)
	print("Length of axis: {0}".format(axs.shape))
	
	total_epochs = metrics[0].shape[0]
	x_values = np.linspace(1, total_epochs, num=total_epochs, dtype=np.int32)
	
	for m, l in zip(metrics, lbs):
		for i in range(0, len(axes)):
			ax = axs[i]
			axis = axes[i]
			ax.plot(x_values, m[:, axis], linewidth=2, label=l)
			ax.set(xlabel=xlb[i], ylabel=ylb[i], title=titles[i])
			ax.xaxis.set_ticks(np.linspace(1, total_epochs, num=int(total_epochs / epochs_interval), dtype=np.int32))
			ax.legend(loc='center right')
	
	plt.suptitle(fig_title, fontsize=14, fontweight='bold')
	plt.show()


# 图像预处理函数
def preprocess_features(imgs):
	# gray scale
	X_grayscale = np.asarray(list(map(lambda img: to_grayscale(img), imgs)))
	# histogram equalization
	clahe = cv2.createCLAHE(tileGridSize=(2, 2), clipLimit=15.0)
	X_grayscale_equalized = np.asarray(
		list(map(lambda img: clahe.apply(np.reshape(img.astype(np.uint8), (32, 32))), X_grayscale)))
	# normalised
	X_grayscale_normalised = normalise_images(X_grayscale_equalized, X_grayscale_equalized)
	# reshape to (?, 32, 32 ,1)
	X_grayscale_normalised = np.reshape(X_grayscale_normalised, (X_grayscale_normalised.shape[0], 32, 32, 1))
	return X_grayscale_normalised


def normalise_images(imgs, dist):
	std = np.std(dist)
	# std = 128
	mean = np.mean(dist)
	# mean = 128
	return (imgs - mean) / std


def to_grayscale(img):
	return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)


# 自定义训练集函数
def custom_images():
	training_file = r'D:\PyCharm 2018.3.2\SBTraining'
	testing_file = r'D:\PyCharm 2018.3.2\SBTesting'
	validing_file = r'D:\PyCharm 2018.3.2\SBValiding'
	train_folder = os.listdir(training_file)
	test_folder = os.listdir(testing_file)
	valid_folder = os.listdir(validing_file)
	X_train = []
	y_train = []
	X_test = []
	y_test = []
	X_valid = []
	y_valid = []
	for _folder in train_folder:
		_imgs = os.path.join(training_file, _folder)
		label_index = int(_folder)
		for _img in os.listdir(_imgs):
			if not _img.endswith("ppm") and not _img.endswith("jpg"):
				continue
			_img_path = os.path.join(training_file, _folder, _img)
			img = cv2.imread(_img_path)
			if img is None:
				continue
			if img.shape != (32, 32, 3):
				img = transform.resize(img, (32, 32), mode="constant")
			X_train.append(img)
			y_train.append(label_index)
		print("train %s loaded %d" % (_folder, len(y_train)))
	for _folder in test_folder:
		_imgs = os.path.join(testing_file, _folder)
		label_index = int(_folder)
		for _img in os.listdir(_imgs):
			if not _img.endswith("ppm") and not _img.endswith("jpg"):
				continue
			_img_path = os.path.join(testing_file, _folder, _img)
			img = cv2.imread(_img_path)
			if img is None:
				continue
			if img.shape != (32, 32, 3):
				img = transform.resize(img, (32, 32), mode="constant")
			X_test.append(img)
			y_test.append(label_index)
		print("test %s loaded %d" % (_folder, len(y_test)))
	for _folder in valid_folder:
		_imgs = os.path.join(validing_file, _folder)
		label_index = int(_folder)
		for _img in os.listdir(_imgs):
			if not _img.endswith("ppm") and not _img.endswith("jpg"):
				continue
			_img_path = os.path.join(validing_file, _folder, _img)
			img = cv2.imread(_img_path)
			if img is None:
				continue
			if img.shape != (32, 32, 3):
				img = transform.resize(img, (32, 32), mode="constant")
			X_valid.append(img)
			y_valid.append(label_index)
		print("valid %s loaded %d" % (_folder, len(y_test)))
	X_test = np.array(X_test)
	y_test = np.array(y_test)
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	X_valid = np.array(X_valid)
	y_valid = np.array(y_valid)
	
	with open(r'D:\PyCharm 2018.3.2\SBTraining.p', "wb") as f:
		pickle.dump({"features": X_train, "labels": y_train}, f, protocol=pickle.HIGHEST_PROTOCOL)
	with open(r'D:\PyCharm 2018.3.2\SBTesting.p', "wb") as f:
		pickle.dump({"features": X_test, "labels": y_test}, f, protocol=pickle.HIGHEST_PROTOCOL)
	with open(r'D:\PyCharm 2018.3.2\SBValiding.p', "wb") as f:
		pickle.dump({"features": X_valid, "labels": y_valid}, f, protocol=pickle.HIGHEST_PROTOCOL)
	
	print("pickled successfully")


def group_img_id_to_lbl(lbs_ids, lbs_names):
	"""
	Utility function to group images by label
	"""
	arr_map = []
	for i in range(0, lbs_ids.shape[0]):
		label_id = lbs_ids[i]
		label_name = lbs_names[lbs_names["ClassId"] == label_id]["SignName"].values[0]
		arr_map.append({"img_id": i, "label_id": label_id, "label_name": label_name})
	
	return pd.DataFrame(arr_map)


def group_img_id_to_lb_count(img_id_to_lb):
	"""
	Returns a pivot table table indexed by label id and label name, where the aggregate function is count
	"""
	return pd.pivot_table(img_id_to_lb, index=["label_id", "label_name"], values=["img_id"], aggfunc='count')


def show_image_list(img_list, img_labels, title, cols=2, fig_size=(15, 15), show_ticks=True):
	"""
	Utility function to show us a list of traffic sign images
	"""
	img_count = len(img_list)
	rows = img_count // cols
	cmap = None
	
	fig, axes = plt.subplots(rows, cols, figsize=fig_size)
	
	for i in range(0, img_count):
		img_name = img_labels[i]
		img = img_list[i]
		if len(img.shape) < 3 or img.shape[-1] < 3:
			cmap = "gray"
			img = np.reshape(img, (img.shape[0], img.shape[1]))
		
		if not show_ticks:
			axes[i].axis("off")
		
		axes[i].imshow(img, cmap=cmap)
	
	fig.suptitle(title, fontsize=12, fontweight='bold', y=0.6)
	fig.tight_layout()
	plt.show()
	
	return


def show_random_dataset_images(group_label, imgs, to_show=5):
	"""
	This function takes a DataFrame of items group by labels as well as a set of images and randomly selects to_show images to display
	"""
	for (lid, lbl), group in group_label:
		# print("[{0}] : {1}".format(lid, lbl))
		rand_idx = np.random.randint(0, high=group['img_id'].size, size=to_show, dtype='int')
		selected_rows = group.iloc[rand_idx]
		
		selected_img = list(map(lambda img_id: imgs[img_id], selected_rows['img_id']))
		selected_labels = list(map(lambda label_id: label_id, selected_rows['label_id']))
		show_image_list(selected_img, selected_labels, "{0}: {1}".format(lid, lbl), cols=to_show, fig_size=(7, 7),
		                show_ticks=False)


if __name__ == "__main__":
	custom_images()
