import healpy as hp
import numpy as np
from astropy.table import Table
from SkyTools import healpixhelper as myhp
from SkyTools import coordhelper, masking
import sys
import pymaster as nmt
import os
import glob
from functools import partial
from scipy import stats


lensing_dir = '/home/graysonpetter/ssd/Dartmouth/data/lensing_maps/'

def make_master_workspace(ellbins, mask1, mask2=None, apodize=None, writename=None):
	lowedges, highedges = ellbins[:-1], (ellbins - 1)[1:]
	w = nmt.NmtWorkspace()
	b = nmt.NmtBin.from_edges(lowedges, highedges)
	if apodize is not None:
		print('apodizing lensing mask')
		mask2 = nmt.mask_apodization(mask2, apodize, apotype='C1')

	mask1_field = nmt.NmtField(mask1, [np.zeros(len(mask1))])
	if mask2 is not None:
		mask2_field = nmt.NmtField(mask2, [np.zeros(len(mask2))])
		w.compute_coupling_matrix(mask1_field, mask2_field, b)
	else:
		w.compute_coupling_matrix(mask1_field, mask1_field, b)
	if writename is not None:
		w.write_to('%s.fits' % writename)
	return w


def unmasked_sky_fraction(mask1, mask2=None):
	if mask2 is None:
		return np.sum(mask1) / len(mask1)
	else:
		return np.sum(mask1 * mask2) / len(mask1)

def density_map(lons, lats, mask, weights):
	nside = hp.npix2nside(len(mask))
	data_density = np.array(myhp.healpix_density_map(lons, lats, nside, weights)).astype(np.float32)
	data_density[np.where(np.logical_not(mask))] = np.nan

	return data_density


def density_contrast_map(lons, lats, mask, weights):
	density = density_map(lons, lats, mask, weights)

	meandensity = np.nanmean(density)

	contrast = (density - meandensity) / meandensity

	contrast[np.isnan(contrast) | np.logical_not(np.isfinite(contrast))] = 0.

	return contrast


def measure_power_spectrum(ell_bins, map1, mask1, map2=None, mask2=None, master_workspace=None):

	if master_workspace is not None:
		w = master_workspace
		field1 = nmt.NmtField(mask1, [map1])
		if map2 is None:
			field2 = field1
		else:
			field2 = nmt.NmtField(mask2, [map2])
		cl = w.decouple_cell(nmt.compute_coupled_cell(field1, field2))[0]
	else:
		skyfrac = unmasked_sky_fraction(mask1, mask2)
		if map2 is None:
			cl = hp.anafast(map1, lmax=np.max(ell_bins)) / skyfrac
		else:
			cl = hp.anafast(map1, map2=map2, lmax=np.max(ell_bins)) / skyfrac

		cl = stats.binned_statistic(np.arange(1, np.max(ell_bins)+2), cl, bins=ell_bins, statistic='median')[0]
	return cl


def measure_xcorr(ell_bins, map1, mask1, map2=None, mask2=None,
				  apodize=None, accurate=True,
				  noisemap_filenames=None, n_realizations=0):
	ell_bins = np.array(ell_bins).astype(int)
	lowedges, highedges = ell_bins[:-1], (ell_bins - 1)[1:]
	b = nmt.NmtBin.from_edges(lowedges, highedges)
	eff_ells = b.get_effective_ells()
	if accurate:
		w = make_master_workspace(ell_bins, mask1, mask2, apodize)
		xcorr = measure_power_spectrum(ell_bins, map1, mask1, map2, mask2, master_workspace=w)
	else:
		xcorr = measure_power_spectrum(ell_bins, map1, mask1, map2, mask2, master_workspace=None)
	output = {'ell_bins': ell_bins, 'ell': eff_ells, 'cl': xcorr}

	if noisemap_filenames is not None:
		realizations = []
		for name in noisemap_filenames:
			noisemap = hp.read_map(name)
			realizations.append(measure_power_spectrum(ell_bins, map1, mask1, noisemap, mask2, master_workspace=None))
		output['cl_err'] = np.std(realizations, axis=0)
	return output

def measure_planck_xcorr(ell_bins, equatorial_coords, nside, equatorial_randcoords=None,
						 mask=None, weights=None, accurate=True, n_noisemaps=30):

	planckmap = hp.read_map(lensing_dir + "Planck18/derived/%s/unsmoothed.fits" % nside)
	planckmask = hp.read_map(lensing_dir + "Planck18/derived/%s/mask.fits" % nside)

	if mask is None:
		randl, randb = coordhelper.equatorial_to_galactic(equatorial_randcoords[0], equatorial_randcoords[1])
		mask, foo = masking.mask_from_randoms(nside_out=hp.npix2nside(len(planckmask)), randlons=randl, randlats=randb)

	if len(mask) != len(planckmask):
		mask = myhp.proper_ud_grade_mask(mask, newnside=hp.npix2nside(len(planckmask)))

	ls, bs = coordhelper.equatorial_to_galactic(equatorial_coords[0], equatorial_coords[1])
	dcmap = density_contrast_map(ls, bs, mask=mask, weights=weights)

	noisenames = glob.glob(lensing_dir + "Planck18/noise/maps/%s/*.fits" % nside)[:n_noisemaps]

	return measure_xcorr(ell_bins, dcmap, mask, planckmap, planckmask,
						 apodize=0.1, accurate=accurate, noisemap_filenames=noisenames)




