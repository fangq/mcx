"""
author: Shih-Cheng Tu
email: mrtoastcheng@gmail.com

"""
from struct import unpack
import numpy as np 


def load_mch(path):
	"""
	input: 
		path: 
			the file path of the .mch file.

	output: 
		mch_data: 
			the output detected photon data array
			data has at least M*2+2 columns (M=header.maxmedia), the first column is the 
			ID of the detector; columns 2 to M+1 store the number of 
			scattering events for every tissue region; the following M
			columns are the partial path lengths (in mm) for each medium type;
			the last column is the initial weight at launch time of each detecetd
			photon; when the momentum transfer is recorded, M columns of
			momentum tranfer for each medium is inserted after the partial path;
			when the exit photon position/dir are recorded, 6 additional columns
			are inserted before the last column, first 3 columns represent the
			exiting position (x/y/z); the next 3 columns are the dir vector (vx/vy/vz).
			in other words, data is stored in the follow format
			[detid(1) nscat(M) ppath(M) mom(M) p(3) v(3) w0(1)]
		
		header: 
			file header info, a dictionary that contains
			version,medianum,detnum,recordnum,totalphoton,detectedphoton,
			savedphoton,lengthunit,seed byte,normalize,respin]

		photonseed: 
			(optional) if the mch file contains a seed section, this
			returns the seed data for each detected photon. Each row of 
			photonseed is a byte array, which can be used to initialize a  
			seeded simulation. Note that the seed is RNG specific. You must use
			the an identical RNG to utilize these seeds for a new simulation.

	"""
	f = open(path, 'rb')
	mch_data = []
	header = []
	photon_seed = []
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
			respin = unpack('i', f.read(4))[0]
			junk = unpack('4i', f.read(4*4))

			assert version == 1, "version higher than 1 is not supported"


			# data = unpack('%df' % (colcount), f.read(4*colcount))
			data = unpack('%df' % (colcount*saved_photon), f.read(4*colcount*saved_photon))
			data = np.asarray(data).reshape(saved_photon, colcount)
			# number of vortex -> actual length
			data[:, 2:1+maxmedia] = data[:, 2:1+maxmedia] * unitmm
			mch_data.append(data)

			# if "save photon seed" is True
			if seed_byte > 0:

				seeds = unpack('%dB' % (saved_photon*seed_byte), f.read(saved_photon*seed_byte))
				photon_seed.append(np.asarray(seeds).reshape((seed_byte,saved_photon), order='F'))

			if respin > 1:
				total_photon *= respin

			header = {"version": version,
					  "maxmedia": maxmedia,
					  "detnum": detnum,
					  "colcount": colcount,
					  "total_photon": total_photon,
					  "detected": detected,
					  "saved_photon": saved_photon,
					  "unitmm": unitmm,
					  "seed_byte": seed_byte,
					  "normalize": normalize,
					  "respin": respin
					  }

	finally:
		f.close()
	
	mch_data = np.asarray(mch_data).squeeze()

	if seed_byte > 0:
		photon_seed = np.asarray(photon_seed).transpose((0,2,1)).squeeze()
		return mch_data, header, photon_seed
	else:
		return mch_data, header


def load_mc2(path, dimension):
	"""
	input: 
		path: 
			the file path of the .mc2 file.

		dimension: 
			an array to specify the output data dimension
			normally, dim=[nx,ny,nz,nt]

	output: 
		data: 
			the output MCX solution data array, in the
			same dimension specified by dim

	"""
	f = open(path, 'rb')
	data = f.read()
	data = unpack('%df' % (len(data)/4), data)
	data = np.asarray(data).reshape(dimension, order='F')

	return data
