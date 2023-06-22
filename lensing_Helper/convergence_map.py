import numpy as np
import healpy as hp
import pandas as pd
import glob
import astropy.units as u
from . import params
paramobj = params.param_obj()
datadir = paramobj.data_dir


# take a binary mask and properly downgrade it to a lower resolution, expanding the mask to cover all pixels which
# touch bad pixels in the high res map
def downgrade_mask(mask, newnside):
	mask_lowres_proper = hp.ud_grade(mask.astype(float), nside_out=newnside).astype(float)
	mask_lowres_proper = np.where(mask_lowres_proper == 1., True, False).astype(bool)
	return mask_lowres_proper

def rotate_mask(mask, coords=['C', 'G'], conservative=True):
	mask = np.array(mask, dtype=bool)
	transform = hp.Rotator(coord=coords)
	# https://stackoverflow.com/questions/68010539/healpy-rotate-a-mask-
	# together-with-the-map-in-hp-ma-vs-separately-produce-di#
	m = hp.ma(np.arange(len(mask), dtype=np.float32))
	m.mask = mask

	# if you use a float mask and rotate, healpix interpolates near border
	if conservative:
		rotated_mask = transform.rotate_map_pixel(m.mask)
		# round the interpolated values to 0 or 1
		return np.around(rotated_mask)
	# otherwise, healpix just rotates the binary mask without interpolation, might be unsafe
	else:
		return transform.rotate_map_pixel(m.mask)



# take in healpix map which defaults to using the UNSEEN value to denote masked pixels and return
# a masked map with NaNs instead
def set_unseen_to_nan(map):
	map[np.where(np.logical_or(map == hp.UNSEEN, np.logical_and(map < -1e30, map > -1e31)))] = np.nan
	return map


# convert a NaN scheme masked map back to the UNSEEN scheme for healpix manipulation
def set_nan_to_unseen(map):
	map[np.isnan(map)] = hp.UNSEEN
	return map


# convert UNSEEN scheme masked map to native numpy masked scheme
def set_unseen_to_mask(map):
	x = np.ma.masked_where(map == hp.UNSEEN, map)
	x.fill_value = hp.UNSEEN
	return x


# zeroes out alm amplitudes for less than a maximum l cutoff and above a maximum cutoff
def zero_modes(almarr, lmin_cut, lmax_cut):
	lmax = hp.Alm.getlmax(len(almarr))
	l, m = hp.Alm.getlm(lmax=lmax)
	almarr[np.where(l < lmin_cut)] = 0.0j
	almarr[np.where(l > lmax_cut)] = 0.0j
	return almarr


# smooth an alm with Wiener filter which inversely weights modes by their noise
def wiener_filter(almarr):
	lmax = hp.Alm.getlmax(len(almarr))
	l, m = hp.Alm.getlm(lmax=lmax)

	noise_table = pd.read_csv('maps/nlkk.dat', delim_whitespace=True, header=None)
	cl_plus_nl = np.array(noise_table[2])
	nl = np.array(noise_table[1])
	cl = cl_plus_nl - nl

	wien_factor = cl/cl_plus_nl

	almarr = hp.smoothalm(almarr, beam_window=wien_factor)
	return almarr


# apply smoothing to a map with masking
def masked_smoothing(inmap, rad=5.0):
	inmap[np.where(inmap == hp.UNSEEN)] = np.nan
	copymap = inmap.copy()
	copymap[inmap != inmap] = 0
	smooth = hp.smoothing(copymap, fwhm=np.radians(rad))
	mask = 0 * inmap.copy() + 1
	mask[inmap != inmap] = 0
	smoothmask = hp.smoothing(mask, fwhm=np.radians(rad))
	final = smooth / smoothmask
	final[np.where(np.isnan(final))] = hp.UNSEEN
	return final


# read in a klm fits lensing convergence map, zero l modes desired, write out map
def klm_2_map(klmname, mapname, nsides):
	# read in planck alm convergence data
	planck_lensing_alm = hp.read_alm(klmname)
	filtered_alm = zero_modes(planck_lensing_alm, 100)
	# generate map from alm data
	planck_lensing_map = hp.sphtfunc.alm2map(filtered_alm, nsides, lmax=4096)
	hp.write_map(mapname, planck_lensing_map, overwrite=True, dtype=float)


# smooth map with gaussian of fwhm = width arcminutes
def smooth_map(mapname, width, outname):
	map = hp.read_map(mapname)
	fwhm = width/60.*np.pi/180.
	smoothed_map = hp.sphtfunc.smoothing(map, fwhm=fwhm)

	hp.write_map(outname, smoothed_map, overwrite=True, dtype=float)


# mask map and remove the mean field if desired
def mask_map(map, mask, outmap):
	# read in map and mask
	importmap = hp.read_map(map)
	importmask = hp.read_map(mask).astype(np.bool)
	# set mask, invert
	masked_map = hp.ma(importmap)
	masked_map.mask = np.logical_not(importmask)
	masked_map = masked_map.filled()

	hp.write_map(outmap, masked_map, overwrite=True, dtype=float)


# input klm file and output final smoothed, masked map for analysis
def klm_2_product(klmname, width, nsides, lmin, maskname=None, lmax=None, coord=None, subtract_mf=False,
				  writename=None):

	# read in planck alm convergence data
	planck_lensing_alm = hp.read_alm(klmname)

	if lmax is None:
		# trying to generate map from l modes >~ 2*nside causes ringing
		lmax_cut = 2 * nsides
	else:
		lmax_cut = lmax

	# if you're going to transform coordinates, usually from equatorial to galactic
	if coord is not None:
		r = hp.Rotator(coord=[coord, 'G'])
		planck_lensing_alm = r.rotate_alm(planck_lensing_alm)

	lmax_fixed = hp.Alm.getlmax(len(planck_lensing_alm))

	if subtract_mf:
		mf_alm = hp.read_alm('maps/mf_klm.fits')
		planck_lensing_alm = planck_lensing_alm - mf_alm

	# if you want to smooth with a gaussian
	if width > 0:
		# transform a gaussian of FWHM=width in real space to harmonic space
		k_space_gauss_beam = hp.gauss_beam(fwhm=width.to('radian').value, lmax=lmax_fixed)
		# if truncating small l modes
		if lmin > 0:
			# zero out small l modes in k-space filter
			k_space_gauss_beam[:lmin] = 0

		# smooth in harmonic space
		filtered_alm = hp.smoothalm(planck_lensing_alm, beam_window=k_space_gauss_beam)
	else:
		# if not smoothing with gaussian, just remove small l modes
		filtered_alm = zero_modes(planck_lensing_alm, lmin, lmax_cut)

	planck_lensing_map = hp.sphtfunc.alm2map(filtered_alm, nsides, lmax=lmax_fixed)

	if maskname is not None:
		# mask map
		importmask = hp.read_map(maskname)
		mask_nside = hp.npix2nside(len(importmask))
		if nsides != mask_nside:

			finalmask = downgrade_mask(importmask, nsides)
		else:
			finalmask = importmask.astype(np.bool)

		# set mask, invert
		smoothed_masked_map = hp.ma(planck_lensing_map)
		smoothed_masked_map.mask = np.logical_not(finalmask)
		if writename:
			hp.write_map('%s.fits' % writename, smoothed_masked_map.filled(), overwrite=True, dtype=float)
	else:
		if writename:
			hp.write_map('%s.fits' % writename, planck_lensing_map, overwrite=True, dtype=float)



def write_planck_maps(real, noise, width, nsides, lmin, lmax=None):
	planckmask = hp.read_map(datadir + 'lensing_maps/Planck18/raw/mask_2048.fits')
	# apodization will be handled by pymaster, so just downgrade binary mask to correct resolution
	if hp.npix2nside(len(planckmask)) != nsides:
		planckmask = downgrade_mask(planckmask, newnside=nsides)
	if lmax is None:
		lmax = nsides * 2

	hp.write_map(datadir + 'lensing_maps/Planck18/derived/%s/mask.fits' % nsides, planckmask, overwrite=True, dtype=float)
	if real:
		klm_2_product(klmname=datadir + 'lensing_maps/Planck18/raw/dat_klm.fits', width=0*u.arcmin,
					  maskname=None, lmax=lmax,
					  nsides=nsides, lmin=lmin,
					  writename=datadir + 'lensing_maps/Planck18/derived/%s/unsmoothed' % nsides)
	if width > 0:
			klm_2_product(klmname=datadir + 'lensing_maps/Planck18/raw/dat_klm.fits', width=width*u.arcmin,
						  maskname=datadir + 'lensing_maps/Planck18/derived/%s/mask.fits' % nsides,
						  nsides=nsides, lmin=lmin,
						  writename=datadir + 'lensing_maps/Planck18/derived/%s/smoothed_masked' % nsides)
	if noise:
		realsnames = glob.glob(datadir + 'lensing_maps/Planck18/noise/klms/sim*')
		for j in range(len(realsnames)):
			klm_2_product(realsnames[j], width=0*u.arcmin, maskname=None, nsides=nsides, lmax=lmax,
						  lmin=lmin, writename=datadir + 'lensing_maps/Planck18/noise/maps/%s/%s' % (nsides, j))


def write_planck_npipe_maps(width, nsides, lmin, lmax):
	planckmask = hp.read_map(datadir + 'lensing_maps/Planck22_PR3_like/raw/mask.fits.gz')
	# apodization will be handled by pymaster, so just downgrade binary mask to correct resolution
	if hp.npix2nside(len(planckmask)) != nsides:
		planckmask = downgrade_mask(planckmask, newnside=nsides)

	hp.write_map(datadir + 'lensing_maps/Planck22_PR3_like/derived/mask.fits', planckmask, overwrite=True, dtype=float)

	klm_2_product(klmname=datadir + 'lensing_maps/Planck22_PR3_like/raw/dat_MV_klm.fits', width=0*u.arcmin,
				  maskname=None, lmax=lmax,
				  nsides=nsides, lmin=lmin,
				  writename=datadir + 'lensing_maps/Planck22_PR3_like/derived/unsmoothed')
	if width > 0:
		klm_2_product(klmname=datadir + 'lensing_maps/Planck22_PR3_like/raw/dat_MV_klm.fits', width=width*u.arcmin,
					  maskname=datadir + 'lensing_maps/Planck22_PR3_like/derived/mask.fits',
					  nsides=nsides, lmin=lmin,
					  writename=datadir + 'lensing_maps/Planck22_PR3_like/derived/smoothed_masked')


def write_spt_maps(width, nsides, lmin, lmax, noise=True):
	sptmask = hp.read_map(datadir + 'lensing_maps/SPT17/raw/mask4096.fits')

	# rotate SPT mask to Galactic
	transform = hp.Rotator(coord=['C', 'G'])
	spt_gal_mask = transform.rotate_map_pixel(sptmask)
	# turn into binary mask
	spt_gal_mask[np.where(spt_gal_mask < 0.5)] = 0
	spt_gal_mask[np.where(spt_gal_mask > 0.5)] = 1
	# downgrade to correct resolution
	spt_gal_mask = downgrade_mask(spt_gal_mask, newnside=nsides)

	hp.write_map(datadir + 'lensing_maps/SPT17/derived/mask.fits', spt_gal_mask, overwrite=True, dtype=float)


	klm_2_product(klmname=datadir + 'lensing_maps/SPT17/raw/spt_gal.alm', width=0 * u.arcmin,
				  maskname=None,
				  nsides=nsides, lmin=lmin, lmax=lmax, writename=datadir + 'lensing_maps/SPT17/derived/unsmoothed')
	klm_2_product(klmname=datadir + 'lensing_maps/SPT17/raw/spt_gal.alm',
				  width=width*u.arcmin,
				  maskname=datadir + 'lensing_maps/SPT17/derived/mask.fits',
				  nsides=nsides, lmin=lmin, writename=datadir + 'lensing_maps/SPT17/derived/smoothed_masked')


	if noise:
		realsnames = sorted(glob.glob(datadir + 'lensing_maps/SPT17/noise/klms/sim*'))
		for j in range(len(realsnames)):
			klm_2_product(realsnames[j], width=0*u.arcmin, maskname=None, nsides=nsides, lmax=lmax,
						  lmin=lmin, writename=datadir + 'lensing_maps/SPT17/noise/maps/%s' % j)





def weak_lensing_map(tomo_bin='tomo4', reconstruction='wiener', width=0*u.arcmin):
	"""k_map = Table.read('lensing_maps/desy1/y1a1_spt_im3shape_0.9_1.3_kE.fits')['kE']
	mask = Table.read('lensing_maps/desy1/y1a1_spt_im3shape_0.9_1.3_mask.fits')['mask']

	smoothed = hp.smoothing(k_map, width.to('radian').value)
	smoothed_masked_map = hp.ma(smoothed)
	smoothed_masked_map.mask = np.logical_not(mask)

	hp.write_map('lensing_maps/desy1/smoothed_masked.fits', smoothed_masked_map.filled(), overwrite=True,
				 dtype=np.single)"""
	lensmap = hp.read_map('lensing_maps/desy3/%s_%s.fits' % (reconstruction, tomo_bin))
	desmask = hp.read_map('lensing_maps/desy3/glimpse_mask.fits')
	lensmap[np.where(np.logical_not(desmask))] = hp.UNSEEN
	hp.write_map('lensing_maps/desy3/smoothed_masked.fits', lensmap, overwrite=True, dtype=float)




def ACT_map(nside, lmin, lmax, smoothfwhm=None):
	from pixell import enmap, reproject, utils
	import healpixhelper
	bnlensing = enmap.read_map('lensing_maps/ACT/act_planck_dr4.01_s14s15_BN_lensing_kappa_baseline.fits')
	bnmask = enmap.read_map('lensing_maps/ACT/act_dr4.01_s14s15_BN_lensing_mask.fits')
	wc_bn_mean = np.mean(np.array(bnmask) ** 2)
	bnlensing = bnlensing * wc_bn_mean

	bnlensing_hp = reproject.healpix_from_enmap(bnlensing, lmax=lmax, nside=nside)
	bnlensing_alm = hp.map2alm(bnlensing_hp, lmax=lmax)
	bnlensing_alm = zero_modes(bnlensing_alm, lmin_cut=lmin, lmax_cut=lmax)
	r = hp.Rotator(coord=['C', 'G'])
	bnlensing_alm = r.rotate_alm(bnlensing_alm)

	bnlensing_hp = hp.alm2map(bnlensing_alm, nside, lmax)

	bnmask_hpx = reproject.healpix_from_enmap(bnmask, lmax=lmax, nside=nside)
	bnmask_gal = healpixhelper.change_coord(bnmask_hpx, ['C', 'G'])
	bnmask_gal[np.where(bnmask_gal > 0.8)] = 1
	bnmask_gal[np.where(bnmask_gal <= 0.8)] = 0

	hp.write_map('lensing_maps/ACT_BN/unsmoothed.fits', bnlensing_hp, overwrite=True)
	hp.write_map('lensing_maps/ACT_BN/mask.fits', bnmask_gal, overwrite=True)


	#wc_bn_mean = np.mean(wc_bn**2)
	#bnlensing_hp = bnlensing_hp * wc_bn_mean



	#smoothbn = hp.smoothing(bnlensing_hp, fwhm=(smoothfwhm * u.arcmin.to('rad')))


	#smoothbn = healpixhelper.change_coord(smoothbn, ['C', 'G'])

	#smoothbn[np.where(wc_bn < 0.8)] = hp.UNSEEN

	d56lensing = enmap.read_map('lensing_maps/ACT/act_planck_dr4.01_s14s15_D56_lensing_kappa_baseline.fits')
	d56mask = enmap.read_map('lensing_maps/ACT/act_dr4.01_s14s15_D56_lensing_mask.fits')
	wc_d56_mean = np.mean(np.array(d56mask) ** 2)
	d56lensing = d56lensing * wc_d56_mean

	d56lensing_hp = reproject.healpix_from_enmap(d56lensing, lmax=lmax, nside=nside)
	d56lensing_alm = hp.map2alm(d56lensing_hp, lmax=lmax)
	d56lensing_alm = zero_modes(d56lensing_alm, lmin_cut=lmin, lmax_cut=lmax)
	r = hp.Rotator(coord=['C', 'G'])
	d56lensing_alm = r.rotate_alm(d56lensing_alm)

	d56lensing_hp = hp.alm2map(d56lensing_alm, nside, lmax)

	d56mask_hpx = reproject.healpix_from_enmap(d56mask, lmax=lmax, nside=nside)
	d56mask_gal = healpixhelper.change_coord(d56mask_hpx, ['C', 'G'])
	d56mask_gal[np.where(d56mask_gal > 0.8)] = 1
	d56mask_gal[np.where(d56mask_gal <= 0.8)] = 0

	hp.write_map('lensing_maps/ACT_D56/unsmoothed.fits', d56lensing_hp, overwrite=True)
	hp.write_map('lensing_maps/ACT_D56/mask.fits', d56mask_gal, overwrite=True)

def sptpol_map(width):
	from astropy.io import fits
	from scipy import signal
	from pixell import enmap, reproject, utils
	import astropy.wcs as worldcoordsys
	mvarr = np.load(datadir + 'lensing_maps/SPT19_Wu/raw/mv_map.npy')
	w = worldcoordsys.WCS(naxis=2)
	w.wcs.crval = [0, -59]
	w.wcs.crpix = [1260.5, 660.5]
	w.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
	w.wcs.cdelt = np.array([0.0166667, -0.0166667])
	header = w.to_header()
	flippedarr = np.fliplr(np.flipud(mvarr))
	hdu = fits.PrimaryHDU(flippedarr, header=header)
	hdu.writeto(datadir + 'lensing_maps/SPT19_Wu/derived/mv_tmp.fits', overwrite=True)
	imap = enmap.read_map(datadir + 'lensing_maps/SPT19_Wu/derived/mv_tmp.fits')
	hpmap = imap.to_healpix(nside=4096)
	if width > 0:
		smoothmap = hp.smoothing(hpmap, fwhm=(width * u.arcmin.to('rad')))
	else:
		#smoothmap = hpmap
		ls = np.arange(3000)

		def gaussian(x, mu, sig):
			return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

		cl = gaussian(ls, 1100, 150)
		smoothmap = hp.smoothing(hpmap, beam_window=cl)
	pixra, pixdec = hp.pix2ang(4096, np.arange(hp.nside2npix(4096)), lonlat=True)
	smoothmap[np.where(((pixra > 30) & (pixra < 330)) | (pixdec > -50) | (pixdec < -65))] = hp.UNSEEN
	hp.write_map(datadir + 'lensing_maps/SPT19_Wu/derived/smoothed_masked.fits', smoothmap, overwrite=True)





