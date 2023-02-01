from os.path import expanduser

class param_obj(object):
	def __init__(self):
		home = expanduser("~")
		self.data_dir = home + '/ssd/Dartmouth/data/'
