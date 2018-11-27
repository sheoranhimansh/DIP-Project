from PIL import Image
import sys
from graph import build_graph, segment_graph
from smooth_filter import gaussian_grid, filter_image
from random import random
from numpy import sqrt
from collections import OrderedDict

def diff_rgb(img, x1, y1, x2, y2):
	r = (img[0][x1, y1] - img[0][x2, y2]) ** 2
	g = (img[1][x1, y1] - img[1][x2, y2]) ** 2
	b = (img[2][x1, y1] - img[2][x2, y2]) ** 2
	return sqrt(r + g + b)

def diff_grey(img, x1, y1, x2, y2):
	v = (img[x1, y1] - img[x2, y2]) ** 2
	return sqrt(v)

def threshold(size, const):
	return (const / size)

def generate_image(image_file , forest, width, height):
	random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
	colors = [random_color() for i in range(width*height)]

	img = Image.new('RGB', (width, height))
	im = img.load()
	col = dict()
	for y in range(height):
		for x in range(width):
			comp = forest.find(y * width + x)
			if comp not in col.keys():
				col[comp] = 0
			col[comp]+=1
	image_file = image_file.load()
	temp = 0
	# for key, value in sorted(list(col.items()), key=lambda x:x[1], reverse=True):
	# 	print(key,value)
		
	while(1):
		for key, value in sorted(list(col.items()), key=lambda x:x[1], reverse=True):
			res = key

			#print(res,forest.find(125*width+125))
			for i in range(1,10):
				if(str(forest.find(125*width+i+125+i)) == str(res)):
					temp = 1
					break
			for i in range(1,10):
				if(str(forest.find(125*width-i+125-i)) == str(res)):
					temp = 1
					break
						
		
			if(temp==1):
				break

		if(temp==1):
				break
	
	#print(res)
	for y in range(height):
		for x in range(width):
			comp = forest.find(y * width + x)
			if comp == res:       
				im[x, y] = image_file[y,x]
			else:
				im[x,y] = 0

			#im[x, y] = colors[comp]

	return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)

def main(sigma, neighborhood, K ,min_size ,input_file):
		neighbor = neighborhood
		if neighbor != 4 and neighbor!= 8:
			print('Invalid neighborhood choosed. The acceptable values are 4 or 8.')
			print('Segmenting with 4-neighborhood...')

		image_file = Image.open(input_file)
		image_file = image_file.resize((250,250))

		size = image_file.size
		#print ('Image info: ', image_file.format, size, image_file.mode)

		grid = gaussian_grid(sigma)

		#2print(image_file.mode)
		if image_file.mode == 'RGB':
			image_file.load()
			r, g, b = image_file.split()

			r = filter_image(r, grid)
			g = filter_image(g, grid)
			b = filter_image(b, grid)

			smooth = (r, g, b)
			diff = diff_rgb
		else:
			return 0

		graph = build_graph(smooth, size[1], size[0], diff, neighbor == 8)
		forest = segment_graph(graph, size[0]*size[1], K, min_size, threshold)

		image = generate_image(image_file , forest, size[1], size[0])
		return image

		print ('Number of components: %d' % forest.num_sets)
