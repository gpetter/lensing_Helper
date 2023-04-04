import healpy as hp
import numpy as np
from astropy.table import Table
from SkyTools import healpixhelper as myhp
from SkyTools import coordhelper
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
	output = {'ell': eff_ells, 'cl': xcorr}

	if noisemap_filenames is not None:
		realizations = []
		for name in noisemap_filenames:
			noisemap = hp.read_map(name)
			realizations.append(measure_power_spectrum(ell_bins, map1, mask1, noisemap, mask2, master_workspace=None))
		output['cl_err'] = np.std(realizations, axis=0)
	return output

def measure_planck_xcorr(ell_bins, coords, mask, weights=None, accurate=True, n_noisemaps=30):

	planckmap = hp.read_map(lensing_dir + "Planck18/derived/unsmoothed.fits")
	planckmask = hp.read_map(lensing_dir + "Planck18/derived/mask.fits")
	if len(mask) != len(planckmask):
		mask = myhp.proper_ud_grade_mask(mask, newnside=hp.npix2nside(len(planckmask)))

	ls, bs = coordhelper.equatorial_to_galactic(coords[0], coords[1])
	dcmap = density_contrast_map(ls, bs, mask=mask, weights=weights)

	noisenames = glob.glob(lensing_dir + "Planck18/noise/maps/*.fits")[:n_noisemaps]

	return measure_xcorr(ell_bins, dcmap, mask, planckmap, planckmask,
						 apodize=0.1, accurate=accurate, noisemap_filenames=noisenames)


def xcorr_of_bin(bootnum, dcmap, which_lens, master=True, plot=False):
	mask = hp.read_map('masks/union.fits')
	lensmask = hp.read_map('lensing_maps/%s/mask.fits' % which_lens)


	if bootnum > 0:
		try:
			lensmap = hp.read_map('lensing_maps/%s/noise/maps/%s.fits' % (which_lens, bootnum - 1))
		except:
			print('No lensing noise map found. Will use random realization of data power spectrum (not sure if this '
			      'is valid)')
			datamap = hp.read_map('lensing_maps/%s/unsmoothed.fits' % which_lens)
			datacl = hp.anafast(datamap)
			lensmap = hp.synfast(datacl, nside=hp.npix2nside(len(lensmask)))

	else:
		lensmap = hp.read_map('lensing_maps/%s/unsmoothed.fits' % which_lens)


	if master:
		lensfield = nmt.NmtField(lensmask, [lensmap])

		dcfield = nmt.NmtField(mask, [dcmap])

		if plot:
			plotting.plot_hpx_map(dcfield.get_maps()[0], 'xcorr_maps/masked_contrast')
			plotting.plot_hpx_map(lensfield.get_maps()[0], 'xcorr_maps/masked_lens')

		wsp = nmt.NmtWorkspace()
		wsp.read_from('masks/namaster/%s_workspace.fits' % which_lens)
		cl = wsp.decouple_cell(nmt.compute_coupled_cell(dcfield, lensfield))[0]

		# divide by pixel window function at effective ls of bandpowers
		#scales = np.array(np.load('results/lensing_xcorrs/%s_scales.npy' % which_lens, allow_pickle=True),
		# dtype=int) - 1
		#pixwin = hp.pixwin(nside=hp.npix2nside(len(dcmap)), lmax=3500)
		#cl = cl / pixwin[scales]




	else:
		cl = hp.anafast(dcmap, lensmap, lmax=2048)

	return cl


def xcorr_by_bin(pool, nboots, samplename, lensname, minscale, maxscale, nbins=10, master=True, plot_maps=False,
                 fullsample=False):
	boots = list(np.arange(nboots + 1))


	oldfiles = glob.glob('results/lensing_xcorrs/%s_%s*' % (samplename, lensname))
	for oldfile in oldfiles:
		os.remove(oldfile)




	nside = hp.npix2nside(len(hp.read_map('masks/union.fits')))

	tab = Table.read('catalogs/derived/catwise_binned.fits')
	nbins = int(np.max(tab['bin']))
	del tab

	for j in range(nbins):
		binnedtab = Table.read('catalogs/derived/catwise_binned_%s.fits' % (j+1))
		dcmap = density_contrast_map(binnedtab['RA'], binnedtab['DEC'], nside=nside, weights=binnedtab['weight'],
		                             plot=plot_maps)

		part_func = partial(xcorr_of_bin, which_lens=lensname, dcmap=dcmap, master=master, plot=plot_maps)
		cls = list(pool.map(part_func, boots))


		if master:
			binnedcls = cls
		else:
			scales = np.logspace(np.log10(minscale), np.log10(maxscale), nbins + 1)
			lmodes = np.arange(len(cls[0])) + 1
			idxs = np.digitize(lmodes, scales)
			binnedcls = []
			for k in range(1, nbins+1):
				binnedcls.append(np.nanmean(cls[0][np.where(idxs == k)]))


		np.array(binnedcls).dump('results/lensing_xcorrs/%s_%s_%s.npy' % (samplename, lensname, j + 1))

	pool.close()

def visualize_xcorr():
	planckmap = hp.read_map('lensing_maps/planck/smoothed_masked.fits')
	tab = Table.read('catalogs/derived/catwise_binned.fits')
	dcmap = density_contrast_map(tab['RA'], tab['DEC'], nside=hp.npix2nside(len(planckmap)), weights=tab['weight'])
	smoothdcmap = hp.smoothing(dcmap, fwhm=1*np.pi/180.)
	smoothplanckmap = hp.smoothing(planckmap, fwhm=0.1 * np.pi/180.)


	dcproj = hp.gnomview(smoothdcmap, rot=[0, 80], xsize=2000, return_projected_map=True)
	planckproj = hp.gnomview(smoothplanckmap, rot=[0, 80], xsize=2000, return_projected_map=True)

	plotting.visualize_xcorr(dcproj, planckproj)


if __name__ == "__main__":
	samplename = 'catwise'
	lens_name = 'planck'
	plot_maps = False

	oldfiles = glob.glob('results/lensing_xcorrs/%s_%s*' % (samplename, lens_name))

	for file in oldfiles:
		os.remove(file)

	import schwimmbad
	lmin, lmax, n_l_bins = 75, 1000, 10
	write_master_workspace(lmin, lmax, whichlens=lens_name, nbins=n_l_bins, apodize=0.1)

	# use different executor based on command line arguments
	# lets code run either serially (python measure_clustering.py)
	# or with multiprocessing to do bootstraps in parallel (python measure_clustering.py --ncores=5)
	# or with MPI
	from argparse import ArgumentParser
	parser = ArgumentParser(description="Schwimmbad example.")


	group = parser.add_mutually_exclusive_group()
	group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
	group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
	args = parser.parse_args()

	pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
	if args.mpi:
		if not pool.is_master():
			pool.wait()
			sys.exit(0)



	#visualize_xcorr()
	xcorr_by_bin(pool, nboots=10, samplename=samplename, lensname=lens_name, minscale=lmin, maxscale=lmax,
	             nbins=n_l_bins, master=True, plot_maps=plot_maps)

	if lens_name == 'SPT':
		for j in range(2):
			planckxpower = np.load('results/lensing_xcorrs/%s_%s_%s.npy' % (samplename, 'planck', (j+1)),
			                   allow_pickle=True)
			planckpower = planckxpower[0]
			planckpower_err = np.std(planckxpower[1:], axis=0)
			planckscales = np.load('results/lensing_xcorrs/%s_scales.npy' % 'planck', allow_pickle=True)
			planckscales.dump('results/lensing_xcorrs/planck+SPT_scales.npy')

			sptxpower = np.load('results/lensing_xcorrs/%s_%s_%s.npy' % (samplename, lens_name, (j + 1)),
			                       allow_pickle=True)
			sptpower = sptxpower[0]
			sptpower_err = np.std(sptxpower[1:], axis=0)
			sptscales = np.load('results/lensing_xcorrs/%s_scales.npy' % lens_name, allow_pickle=True)

			bothpower = np.average([planckpower, sptpower], weights=[1/np.square(planckpower_err), 1/np.square(
				sptpower_err)], axis=0)
			botherr = np.sqrt(1 / (1/np.square(planckpower_err) + 1/np.square(sptpower_err)))
			np.array([bothpower, botherr]).dump('results/lensing_xcorrs/%s_%s_%s.npy' % (samplename, 'planck+SPT',
			                                                                          (j + 1)))
	lensing_plots.lensing_xcorrs(samplename, lensnames=['planck'], lcl=True)



