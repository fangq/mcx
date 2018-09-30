from struct import unpack
import numpy as np 

def load_mch(path):
	f = open(path, 'rb')
	mch_data = []
	try:
		while True:
			# the first 4 byte in a mch file is always 'MCXH'
			buffer_ = f.read(4)
			if not buffer_:
				break
			elif buffer_ != b'MCXH':
				raise Exception("It might not be a mch file!")

			version = unpack('i', f.read(4))[0]
			maxmedia = unpack('i', f.read(4))[0]
			detnum = unpack('i', f.read(4))[0]
			colcount = unpack('i', f.read(4))[0]
			total_photon = unpack('i', f.read(4))[0]
			detected = unpack('i', f.read(4))[0]
			saved_photon = unpack('i', f.read(4))[0]

			unitmm = unpack('f', f.read(4))[0]
			seed_byte = unpack('i', f.read(4))[0]
			normalize = unpack('f', f.read(4))[0]
			junk = unpack('5i', f.read(4*5))
			# data = unpack('%df' % (colcount), f.read(4*colcount))
			data = unpack('%df' % (colcount*saved_photon), f.read(4*colcount*saved_photon))
			data = np.asarray(data).reshape(saved_photon, colcount)
			# number of vortex -> actual length
			data[:, 2:1+maxmedia] = data[:, 2:1+maxmedia] * unitmm

	finally:
		f.close()

	return data

def load_mc2(path, dimension):
	f = open(path, 'rb')
	data = f.read()
	data = unpack('%df' % (len(data)/4), data)
	data = np.asarray(data).reshape(dimension, order='F')

	return data