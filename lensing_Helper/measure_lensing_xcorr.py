import healpy as hp
import numpy as np
from astropy.table import Table
import healpixhelper
import sys
import pymaster as nmt
import os
import glob
from functools import partial
from plotscripts import lensing_plots


def union_spt_mask():
	import masking
	mask = hp.read_map('masks/union.fits')
	sptmap = hp.read_map('lensing_maps/SPT/unsmoothed_masked.fits')
	sptnside = hp.npix2nside(len(sptmap))
	upgraded_unionmask = masking.downgrade_mask(mask, sptnside)
	upgraded_unionmask[np.where(sptmap == hp.UNSEEN)] = 0

	return upgraded_unionmask




def write_master_workspace(minl, maxl, whichlens, nbins, apodize=None):
	if len(glob.glob('masks/namaster/%s_workspace.fits' % whichlens)) > 0:
		print('Removing old Master matrix')
		os.remove('masks/namaster/%s_workspace.fits' % whichlens)
	logbins = np.logspace(np.log10(minl), np.log10(maxl), nbins+1).astype(int)
	lowedges, highedges = logbins[:-1], (logbins - 1)[1:]

	quasar_mask = hp.read_map('masks/union.fits')

	lensmask = hp.read_map('lensing_maps/%s/mask.fits' % whichlens)
	#mask = quasar_mask * lensmask
	#mask = quasar_mask

	if apodize is not None:
		print('apodizing lensing mask')
		lensmask = nmt.mask_apodization(lensmask, apodize, apotype='C1')

	q_mask_field = nmt.NmtField(quasar_mask, [np.zeros(len(quasar_mask))])
	lens_mask_field = nmt.NmtField(lensmask, [np.zeros(len(lensmask))])
	b = nmt.NmtBin.from_edges(lowedges, highedges)

	np.array(b.get_effective_ells()).dump('results/lensing_xcorrs/%s_scales.npy' % whichlens)
	w = nmt.NmtWorkspace()
	w.compute_coupling_matrix(q_mask_field, lens_mask_field, b)

	w.write_to('masks/namaster/%s_workspace.fits' % whichlens)


def write_ls_density_mask():
	randomcat = Table.read('catalogs/randoms/ls_randoms/randoms_0-5_coords.fits')
	randras, randdecs = randomcat['RA'], randomcat['DEC']
	randlons, randlats = healpixhelper.equatorial_to_galactic(randras, randdecs)
	randdensity = healpixhelper.healpix_density_map(randlons, randlats, 2048)
	hp.write_map('masks/ls_density.fits', randdensity, overwrite=True)


def density_map(ras, decs, nside, weights, plot=False):
	lons, lats = healpixhelper.equatorial_to_galactic(ras, decs)
	data_density = np.array(healpixhelper.healpix_density_map(lons, lats, nside, weights)).astype(np.float32)

	mask = hp.read_map('masks/union.fits')
	data_density[np.where(np.logical_not(mask))] = np.nan

	if plot:
		print('plotting density map')
		plotting.plot_hpx_map(data_density, 'xcorr_maps/density')

	return data_density


def density_contrast_map(ras, decs, nside, weights, correct_lost_area=False, plot=False):
	density = density_map(ras, decs, nside, weights, plot=plot)

	meandensity = np.nanmean(density)

	contrast = (density - meandensity) / meandensity

	contrast[np.isnan(contrast) | np.logical_not(np.isfinite(contrast))] = 0.

	if plot:
		print('plotting density contrast')
		plotdata = hp.smoothing(contrast, np.pi / 180.)
		mask = hp.read_map('masks/union.fits')
		plotdata[np.where(np.logical_not(mask))] = np.nan

		plotting.plot_hpx_map(plotdata, 'xcorr_maps/contrast')

	return contrast


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



