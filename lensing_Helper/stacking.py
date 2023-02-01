import numpy as np
import healpy as hp
from . import convergence_map
from astropy.io import fits
import importlib
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import fileinput
import sys
from math import ceil
import glob
import time
from functools import partial
from . import params

paramobj = params.param_obj()
datadir = paramobj.data_dir

# convert ras and decs to galactic l, b coordinates
def equatorial_to_galactic(ra, dec):
	ra_decs = SkyCoord(ra, dec, unit='deg', frame='icrs')
	ls = np.array(ra_decs.galactic.l.radian * u.rad.to('deg'))
	bs = np.array(ra_decs.galactic.b.radian * u.rad.to('deg'))
	return ls, bs


# given list of ras, decs, return indices of sources whose centers lie outside the masked region of the lensing map
def get_qsos_outside_mask(nsides, themap, ras, decs):
	ls, bs = equatorial_to_galactic(ras, decs)
	pixels = hp.ang2pix(nsides, ls, bs, lonlat=True)
	idxs = np.where(themap[pixels] != hp.UNSEEN)
	return idxs


# AzimuthalProj.projmap requires a vec2pix function for some reason, so define one where the nsides are fixed
def newvec2pix(x, y, z):
	return hp.vec2pix(nside=4096, x=x, y=y, z=z)


# perform one iteration of a stack
def stack_iteration(current_sum, current_weightsum, new_cutout, weight, prob_weight, imsize):
	# create an image filled with the value of the weight, set weights to zero where the true map is masked
	wmat = np.full((imsize, imsize), weight)
	wmat[np.isnan(new_cutout)] = 0

	# the weights for summing in the denominator are multiplied by the probabilty weight to account
	# for the fact that some sources aren't quasars and contribute no signal to the stack
	wmat_for_sum = wmat * prob_weight
	# the running total sum is the sum from last iteration plus the new cutout
	new_sum = np.nansum([current_sum, new_cutout], axis=0)
	new_weightsum = np.sum([current_weightsum, wmat_for_sum], axis=0)

	return new_sum, new_weightsum


def sum_projections(lon, lat, weights, prob_weights, imsize, reso, inmap, nstack):
	running_sum, weightsum = np.zeros((imsize, imsize)), np.zeros((imsize, imsize))
	for j in range(nstack):
		azproj = hp.projector.AzimuthalProj(rot=[lon[j], lat[j]], xsize=imsize, reso=reso, lamb=True)
		new_im = weights[j] * convergence_map.set_unseen_to_nan(azproj.projmap(inmap, vec2pix_func=newvec2pix))

		running_sum, weightsum = stack_iteration(running_sum, weightsum, new_im, weights[j], prob_weights[j], imsize)

	return running_sum, weightsum


# for parallelization of stacking procedure, this method will stack a "chunk" of the total stack
def stack_chunk(chunksize, nstack, lon, lat, inmap, weighting, prob_weights, imsize, reso, k):
	# if this is the last chunk in the stack, the number of sources in the chunk probably won't be = chunksize
	if (k * chunksize) + chunksize > nstack:
		stepsize = nstack % chunksize
	else:
		stepsize = chunksize
	highidx, lowidx = ((k * chunksize) + stepsize), (k * chunksize)

	totsum, weightsum = sum_projections(lon[lowidx:highidx], lat[lowidx:highidx], weighting[lowidx:highidx],
	                                    prob_weights[lowidx:highidx], imsize, reso, inmap, stepsize)

	return totsum, weightsum


# stack by computing an average iteratively. this method uses little memory but cannot be parallelized
def stack_projections(ras, decs, weights=None, prob_weights=None, imsize=240, outname=None, reso=1.5, inmap=None,
                      nstack=None, mode='normal', chunksize=500):
	# if no weights provided, weights set to one
	if weights is None:
		weights = np.ones(len(ras))
	if prob_weights is None:
		prob_weights = np.ones(len(ras))
	# if no limit to number of stacks provided, stack the entire set
	if nstack is None:
		nstack = len(ras)

	lons, lats = equatorial_to_galactic(ras, decs)

	if mode == 'normal':
		totsum, weightsum = sum_projections(lons, lats, weights, prob_weights, imsize, reso, inmap, nstack)
		finalstack = totsum/weightsum

	else:
		return

	if outname is not None:
		finalstack.dump('%s.npy' % outname)

	return finalstack


def fast_stack(lons, lats, inmap, weights=None, prob_weights=None, nsides=2048, iterations=500, bootstrap=False):
	if weights is None:
		weights = np.ones(len(lons))

	outerkappa = []


	inmap = convergence_map.set_unseen_to_nan(inmap)
	pix = hp.ang2pix(nsides, lons, lats, lonlat=True)
	neighborpix = hp.get_all_neighbours(nsides, pix)

	centerkappa = inmap[pix]
	neighborkappa = np.nanmean(inmap[neighborpix], axis=0)
	centerkappa = np.nanmean([centerkappa, neighborkappa], axis=0)

	weights[np.isnan(centerkappa)] = 0
	centerkappa[np.isnan(centerkappa)] = 0


	if prob_weights is not None:
		true_weights_for_sum = weights * np.array(prob_weights)
		weightsum = np.sum(true_weights_for_sum)
	else:
		weightsum = np.sum(weights)

	centerstack = np.sum(weights * centerkappa) / weightsum


	if iterations > 0:

		for x in range(iterations):
			bootidx = np.random.choice(len(lons), len(lons))
			outerkappa.append(np.nanmean(inmap[hp.ang2pix(nsides, lons[bootidx], lats[bootidx], lonlat=True)]))

		if bootstrap:
			return centerstack, outerkappa
		else:
			return centerstack, np.nanstd(outerkappa)
	else:
		return centerstack

# for a given sky position, choose which pre-calculated sky projection is centered closest to that position
def find_closest_cutout(l, b, fixedls, fixedbs):
	return np.argmin(hp.rotator.angdist([fixedls, fixedbs], [l, b], lonlat=True))


def stack_cutouts(ras, decs, weights, prob_weights, imsize, nstack, outname=None, bootstrap=False):
	# read in previously calculated projections covering large swaths of sky
	projectionlist = glob.glob('planckprojections/*')
	projections = np.array([np.load(filename, allow_pickle=True) for filename in projectionlist])

	if bootstrap:
		bootidxs = np.random.choice(len(ras), len(ras))
		ras, decs, weights, prob_weights = ras[bootidxs], decs[bootidxs], weights[bootidxs], prob_weights[bootidxs]
	# center longitudes/latitudes of projections
	projlons = [int(filename.split('/')[1].split('.')[0].split('_')[0]) for filename in projectionlist]
	projlats = [int(filename.split('/')[1].split('.')[0].split('_')[1]) for filename in projectionlist]

	# healpy projection objects used to create the projections
	# contains methods to convert from angular position of quasar to i,j position in projection
	projector_objects = [hp.projector.AzimuthalProj(rot=[projlons[i], projlats[i]], xsize=5000, reso=1.5, lamb=True)
	                     for i in range(len(projlons))]
	# convert ras and decs to galactic ls, bs
	lon, lat = equatorial_to_galactic(ras, decs)


	running_sum, weightsum = np.zeros((imsize, imsize)), np.zeros((imsize, imsize))
	# for each source
	for k in range(nstack):
		# choose the projection closest to the QSO's position
		cutoutidx = find_closest_cutout(lon[k], lat[k], projlons, projlats)
		cutout_to_use = projections[cutoutidx]
		projobj = projector_objects[cutoutidx]
		# find the i,j coordinates in the projection corresponding to angular positon in sky
		i, j = projobj.xy2ij(projobj.ang2xy(lon[k], lat[k], lonlat=True))
		# make cutout
		cut_at_position = cutout_to_use[int(i-imsize/2):int(i+imsize/2), int(j-imsize/2):int(j+imsize/2)]
		# stack
		running_sum, weightsum = stack_iteration(running_sum, weightsum, cut_at_position, weights[j], prob_weights[j],
		                                         imsize)
	finalstack = running_sum/weightsum
	if outname is not None:
		finalstack.dump('%s.npy' % outname)

	return finalstack


def stack_without_projection():
	cat = Table.read('catalogs/derived/catwise_r90_binned.fits')
	inmap = hp.read_map('lensing_maps/planck/smoothed_masked.fits')
	ras, decs = cat["RA"], cat["DEC"]
	lons, lats = equatorial_to_galactic(ras, decs)
	nside = hp.npix2nside(len(inmap))
	coords = SkyCoord(lons * u.deg, lats * u.deg)
	hpras, hpdecs = hp.pix2ang(nside, np.arange(len(inmap)), lonlat=True)
	st = time.time()
	coordvecs = hp.ang2vec(lons, lats, lonlat=True)

	for j in range(len(lons)):
		pixindisk = hp.query_disc(nside, coordvecs[j], np.pi / 90.)
		hpras, hpdecs = hp.pix2ang(nside, pixindisk, lonlat=True)
		hpcoords = SkyCoord(hpras * u.deg, hpdecs * u.deg)
		separations = coords[j].separation(hpcoords).to('arcmin').value
		if j % 1000 == 0:
			print(time.time() - st)







def clever_stack(stackmap, coords, thetabins):
	import multiprocessing as mp
	maxtheta = np.radians(np.max(thetabins / 60.))


	vecs = hp.ang2vec(coords[0], coords[1], lonlat=True)
	avg_kappas = []

	pixarea_steradian = hp.nside2pixarea(hp.npix2nside(len(stackmap)))
	maxringarea = np.pi * maxtheta ** 2
	pix_in_disc = int(maxringarea / pixarea_steradian)

	chunksize = 10000
	def chunkit(chunk):
		large_disc_idxs = []
		if (chunk * chunksize) + chunksize > len(vecs):
			stepsize = len(vecs) % chunksize
		else:
			stepsize = chunksize
		for j in range(chunk*chunksize, chunk*chunksize+stepsize):
			large_disc_idxs.append(hp.query_disc(hp.npix2nside(len(stackmap)), vecs[j], radius=maxtheta)[:pix_in_disc - 50])
		print(np.shape(large_disc_idxs))
		discvecs = np.transpose(hp.pix2vec(hp.npix2nside(len(stackmap)), large_disc_idxs), axes=[1,0,2])
		print(np.shape(discvecs))
		print(np.shape(vecs[chunk*chunksize:chunk*chunksize+stepsize]))
		kappas = stackmap[np.array(large_disc_idxs)]
		dists = np.degrees(hp.rotator.angdist(vecs[chunk*chunksize:chunk*chunksize+stepsize], discvecs, lonlat=True)) * 60.
		binidxs = np.digitize(dists, bins=thetabins)
		avg_kappas.append(np.bincount(binidxs, weights=kappas) / np.bincount(binidxs))
	ncores = mp.cpu_count()
	nchunks = ceil(len(coords[0]) / chunksize)
	return chunkit(1)
	#p = mp.Pool(ncores)
	#return map(chunkit, np.arange(nchunks))

	#for j in range(len(vecs)):
	#	for k in range(len(thetabins)):
	#		idxs = hp.query_disc(hp.npix2nside(len(stackmap)), vecs[j], radius=thetabins[k])

		#large_disc_idxs.append(idxs)
		#kappas = stackmap[idxs]
		#discvecs = hp.pix2vec(hp.npix2nside(len(stackmap)), idxs)
		#dists = np.degrees(hp.rotator.angdist(vecs[j], discvecs, lonlat=True)) * 60.
		#binidxs = np.digitize(dists, bins=thetabins)
		#avg_kappas.append(np.bincount(binidxs, weights=kappas)/np.bincount(binidxs))
	#discvecs = hp.pix2vec(hp.npix2nside(len(stackmap)), large_disc_idxs)
	#return avg_kappas
		#idxlengths.append(len(idxs))
		#large_disc_idxs.append(idxs)

	#minlength = np.min(idxlengths)


def annulus_stack_chunk(chunk, vecs, chunksize, stackmap, thetas, nside):
	# if current chunk is the last, the stepsize is the remainder number of sources
	if (chunk * chunksize) + chunksize > len(vecs):
		stepsize = len(vecs) % chunksize
	else:
		stepsize = chunksize

	indexbybin = [[] for j in range(len(thetas))]
	# keep track of sum within annuli as well as number of healpixels
	profile, npix = [], []
	# for each source
	for j in range(chunk*chunksize, chunk*chunksize+stepsize):
		# find all pixels within r= maximum bin edge
		bigdisc = hp.query_disc(nside, vecs[j], np.radians(thetas[0]))
		# for each bin
		for k in range(1, len(thetas)):
			# find pixels within next smallest radius
			smalldisc = (hp.query_disc(nside, vecs[j], np.radians(thetas[k])))
			# pixels within outer radius but not within smaller radius are those in the annulus of interest
			indexbybin[k-1] += (np.setdiff1d(bigdisc, smalldisc, assume_unique=True)).tolist()
			# set the current radius to the outer radius for the next iteration
			bigdisc = smalldisc

	# for each bin
	for k in range(len(indexbybin)):
		# index the map using all the indices accumulated by querying annuli around sources
		kappas = stackmap[indexbybin[k]]
		# sum up the pixel values inside annuli around sources
		profile.append(np.nansum(kappas))
		# sum up number of non-masked pixels in annuli around sources
		npix.append(len(np.where(np.isfinite(kappas))))

	return [np.flip(profile), np.flip(npix)]


# stack by using query_disc for speed
def stack_annuli(stackmap, coords, binparams, nthreads=1, chunksize=None):
	# parse map, set mask to NaN
	stackmap = convergence_map.set_unseen_to_nan(stackmap)
	# parse binning parameters
	maxtheta, n_theta_bins = binparams
	# bins are from radius=0 to max_theta
	thetas = np.flip(np.linspace(0, maxtheta, n_theta_bins+1))

	# if chunksize not given, use general rule of number of sources / 20
	if chunksize is None:
		chunksize = int(len(coords[0]) / 20)
		# but if chunksize is too big, set to 2000, works well for 32 GB RAM
		if chunksize > 2000:
			chunksize = 2000

	# number of chunks, round up
	nchunks = ceil(len(coords[0]) / chunksize)
	# convert RAs, decs to vectors
	vecs = hp.ang2vec(coords[0], coords[1], lonlat=True)
	nside = hp.npix2nside(len(stackmap))

	# function which takes chunk number and does stack within that chunk
	partialst = partial(annulus_stack_chunk, vecs=vecs, chunksize=chunksize, stackmap=stackmap, thetas=thetas, nside=nside)

	# multithreading
	if nthreads > 1:
		import multiprocessing as mp
		p = mp.Pool(mp.cpu_count())
		result = list(p.map(partialst, np.arange(nchunks)))
		pix_sum, npix = result[0], result[1]
	else:
		result = list(map(partialst, np.arange(nchunks)))
		pix_sum, npix = result[0], result[1]

	# average within annuli around sources is the total sum of pixel values / the number of pixels
	return np.sum(pix_sum, axis=0) / np.sum(npix, axis=0)