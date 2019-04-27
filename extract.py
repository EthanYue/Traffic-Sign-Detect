import math
from utils import preprocess_features
import numpy as np
import cv2 as cv
from skimage import transform, feature


def preprocess_img(imgBGR, erode_dilate=True):
	"""preprocess the image for contour detection.
	Args:
		imgBGR: source image.
		erode_dilate: erode and dilate or not.
	Return:
		img_bin: a binary image (blue and red).

	"""
	rows, cols, _ = imgBGR.shape
	
	imgHSV = cv.cvtColor(imgBGR, cv.COLOR_BGR2HSV)
	
	Bmin = np.array([100, 43, 46])
	Bmax = np.array([124, 255, 255])
	img_Bbin = cv.inRange(imgHSV, Bmin, Bmax)
	
	Rmin1 = np.array([0, 43, 46])
	Rmax1 = np.array([10, 255, 255])
	img_Rbin1 = cv.inRange(imgHSV, Rmin1, Rmax1)
	
	Rmin2 = np.array([156, 43, 46])
	Rmax2 = np.array([180, 255, 255])
	img_Rbin2 = cv.inRange(imgHSV, Rmin2, Rmax2)
	img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
	img_bin = np.maximum(img_Bbin, img_Rbin)
	
	if erode_dilate is True:
		kernelErosion = np.ones((3, 3), np.uint8)
		kernelDilation = np.ones((3, 3), np.uint8)
		img_bin = cv.erode(img_bin, kernelErosion, iterations=2)
		img_bin = cv.dilate(img_bin, kernelDilation, iterations=2)
	blurred = cv.GaussianBlur(img_bin, (3, 3), 2)
	cannyed = cv.Canny(blurred, 3, 9, 3)
	kernel_sharpen_1 = np.array([
		[-1, -1, -1],
		[-1, 9, -1],
		[-1, -1, -1]])
	img_bin = cv.filter2D(img_bin, -1, kernel_sharpen_1)
	return img_bin


def contour_detect(img_bin, min_area=1000, max_area=-1, wh_ratio=2.0):
	"""detect contours in a binary image.
	Args:
		img_bin: a binary image.
		min_area: the minimum area of the contours detected.
			(default: 0)
		max_area: the maximum area of the contours detected.
			(default: -1, no maximum area limitation)
		wh_ratio: the ration between the large edge and short edge.
			(default: 2.0)
	Return:
		rects: a list of rects enclosing the contours. if no contour is detected, rects=[]
	"""
	rects = []
	_, contours, _ = cv.findContours(img_bin.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	if len(contours) == 0:
		return rects
	max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
	for contour in contours:
		area = cv.contourArea(contour)
		length = cv.arcLength(contour, closed=True)
		if min_area <= area <= max_area:
			x, y, w, h = cv.boundingRect(contour)
			if 0.5 < 1.0 * w / h < wh_ratio:
				# 先用y确定高，再用x确定宽
				_contour = img_bin[y + 2:y + h - 2, x + 2:x + w - 2]
				if filter_circle(_contour):
					C = (4 * math.pi * area) / (length * length)
					# 利用圆度初步对形状进行筛选
					if C > 0.5:
						rects.append([x, y, w, h])
	return rects


def fillHole(img):
	im_floodfill = img.copy()
	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = img.shape[:2]
	mask = np.zeros((h + 2, w + 2), np.uint8)
	
	# Floodfill from po (0, 0)
	cv.floodFill(im_floodfill, mask, (0, 0), 255)
	# Invert floodfilled image
	im_floodfill_inv = cv.bitwise_not(im_floodfill)
	# Combine the two images to get the foreground.
	im_out = img | im_floodfill_inv
	
	return im_out


def filter_circle(img):
	is_circle = False
	h, w = img.shape
	count1 = 0  # 各部分的缺失像素计数器
	count2 = 0
	count3 = 0
	count4 = 0
	# 将img平均分成四份, 进行访问缺失的像素个数、所占比重
	# 先访问左上
	for i in range(0, h // 2):
		for j in range(0, w // 2):
			if img[i, j] == 255:
				count1 += 1
	# 右上
	for i in range(0, h // 2):
		for j in range(w // 2, w):
			if img[i, j] == 255:
				count2 += 1
	
	# 左下
	for i in range(h // 2, h):
		for j in range(0, w // 2):
			if img[i, j] == 255:
				count3 += 1
	
	# 右下
	for i in range(h // 2, h):
		for j in range(w // 2, w):
			if img[i, j] == 255:
				count4 += 1
	
	c1 = count1 / (w * h)  # 左上
	c2 = count2 / (w * h)  # 右上
	c3 = count3 / (w * h)  # 左下
	c4 = count4 / (w * h)  # 右下
	
	# 限定每个比率的差值范围
	if (0.15 < c1 < 0.25) and (0.15 < c2 < 0.25) and (0.15 < c2 < 0.25) and (0.15 < c2 < 0.25):
		# 限制差值, 差值比较容错，相邻块之间差值相近，如左上=右上 and 左下=右下或左上=左下 and 右上=右下
		if (abs(c1 - c2) < 0.06 and abs(c3 - c4) < 0.06) or (abs(c1 - c3) < 0.06 and abs(c2 - c4) < 0.06):
			is_circle = True
	return is_circle


def hog_extra_and_svm_class(proposal, clf, resize=(32, 32)):
	"""classify the region proposal.
	Args:
		proposal: region proposal (numpy array).
		clf: a SVM model.
		resize: resize the region proposal
			(default: (64, 64))
	Return:
		cls_prop: propabality of all classes.
	"""
	img = cv.cvtColor(proposal, cv.COLOR_BGR2GRAY)
	img = cv.resize(img, resize)
	bins = 9
	cell_size = (8, 8)
	cpb = (2, 2)
	norm = "L2"
	features = feature.hog(img, orientations=bins, pixels_per_cell=cell_size, cells_per_block=cpb, block_norm=norm,
	                       transform_sqrt=True)
	features = np.reshape(features, (1, -1))
	cls_prop = clf.predict_proba(features)
	return cls_prop


def extract_image(img_path):
	"""
	:param img_path: 图片路径
	:return: 图片中交通标志的位置，截取到图中的交通标志部分
	"""
	proposal = []
	if img_path.endswith("mp4"):
		cap = cv.VideoCapture("test_video2.mp4")
		cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
		rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
		while (1):
			ret, img = cap.read()
			img_bin = preprocess_img(img, False)
			cv.imshow("bin image", img_bin)
			min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
			rects = contour_detect(img_bin, min_area=min_area)
			img_bbx = img.copy()
			for rect in rects:
				xc = int(rect[0] + rect[2] / 2)
				yc = int(rect[1] + rect[3] / 2)
				
				size = max(rect[2], rect[3])
				x1 = max(0, int(xc - size / 2))
				y1 = max(0, int(yc - size / 2))
				x2 = min(cols, int(xc + size / 2))
				y2 = min(rows, int(yc + size / 2))
				proposal.append(img[y1:y2, x1:x2])
	else:
		img = cv.imread(img_path)
		if img is None:
			return [], []
		rows, cols, _ = img.shape
		# 图片预处理
		img_bin = preprocess_img(img, True)
		# 空洞填充
		img_filled = fillHole(img_bin)
		min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
		# 轮廓监测
		rects = contour_detect(img_filled, min_area=min_area)
		proposal = []
		# 从图片中截取检测到的交通标志
		for rect in rects:
			xc = (rect[0] + rect[2] / 2)
			yc = (rect[1] + rect[3] / 2)
			size = max(rect[2], rect[3])
			x1 = max(0, (xc - size / 2))
			y1 = max(0, (yc - size / 2))
			x2 = min(cols, (xc + size / 2))
			y2 = min(rows, (yc + size / 2))
			_img = img[int(y1)-5:int(y2)+5, int(x1)-5:int(x2)+5]
			_img = cv.resize(_img, (32, 32))
			proposal.append(_img)
	return rects, proposal
