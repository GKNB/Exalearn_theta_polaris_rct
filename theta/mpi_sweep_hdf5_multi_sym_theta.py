#!/usr/bin/env python

import os,sys
sys.path.insert(0,os.path.expanduser("~/g2full_theta/GSASII/"))
import GSASIIscriptable as G2sc

from mpi4py import MPI
import numpy as np

import sweep_utils as su

import time

gpx = []
phase = []


def cubic_lattice(prm):
    """ This function uses a 1D grid.

        Parameters for cubic lattice:
        a = b = c
        alpha = beta = gamma = 90
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[0] # b
    phase['General']['Cell'][3] = prm[0] # c

    # Angles
    phase['General']['Cell'][4] = 90 # alpha
    phase['General']['Cell'][5] = 90 # beta
    phase['General']['Cell'][6] = 90 # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y



def trigonal_lattice(prm):
    """ This function uses a 2D grid.

        Parameters for trigonal lattice:
        a = b = c
        alpha = beta = gamma != 90
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[0] # b
    phase['General']['Cell'][3] = prm[0] # c

    # Angles
    phase['General']['Cell'][4] = prm[1] # alpha
    phase['General']['Cell'][5] = prm[1] # beta
    phase['General']['Cell'][6] = prm[1] # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y



def tetragonal_lattice(prm):
    """ This function uses a 2D grid.

        Parameters for tetragonal lattice:
        a = b != c
        alpha = beta = gamma = 90
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    # c should be differe
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[0] # b
    phase['General']['Cell'][3] = prm[1] # c

    # Angles
    phase['General']['Cell'][4] = 90 # alpha
    phase['General']['Cell'][5] = 90 # beta
    phase['General']['Cell'][6] = 90 # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y



def trigonalHexP_lattice(prm):
    """ This function uses a 2D grid.

        Parameters for trigonal hexP lattice:
        a = b != c
        alpha = beta = 90
        gamma = 120
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    # c should be different
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[0] # b
    phase['General']['Cell'][3] = prm[1] # c

    # Angles
    phase['General']['Cell'][4] = 90 # alpha
    phase['General']['Cell'][5] = 90 # beta
    phase['General']['Cell'][6] = 120 # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y



def orthorhombic_lattice(prm):
    """ This function uses a 3D grid.

        Parameters for orthorhombic lattice:
        a != b != c
        alpha = beta = gamma = 90
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[1] # b
    phase['General']['Cell'][3] = prm[2] # c

    # Angles
    phase['General']['Cell'][4] = 90 # alpha
    phase['General']['Cell'][5] = 90 # beta
    phase['General']['Cell'][6] = 90 # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y



def monoclinic_lattice(prm):
    """ This function uses a 4D grid.

        Parameters for monoclinic lattice:
        a != b != c
        alpha = gamma = 90
        beta != 120
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[1] # b
    phase['General']['Cell'][3] = prm[2] # c

    # Angles
    phase['General']['Cell'][4] = 90      # alpha
    phase['General']['Cell'][5] = prm[3]  # beta
    phase['General']['Cell'][6] = 90      # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y



def triclinic_lattice(prm):
    """ This function uses a 6D grid.

        Parameters for triclinic lattice:
        a != b != c
        alpha != beta != gamma != 90
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[1] # b
    phase['General']['Cell'][3] = prm[2] # c
    
    # Angles
    phase['General']['Cell'][4] = prm[3] # alpha
    phase['General']['Cell'][5] = prm[4] # beta
    phase['General']['Cell'][6] = prm[5] # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y



def main():

    start = time.time()
    if ( len ( sys.argv ) < 2 ) :
        sys.stderr.write ( "\nUsage:  mpi_sweep_triclinic_hdf5.py CONFIG_FILENAME1 CONFIG_FILENAME2 ...\n" )
        sys.exit ( 0 )
 
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        print("Size is ", size)
    print("Rank is ", rank)
   
    for i in range(1, len(sys.argv)):
        conffile = sys.argv[i]
        gParameters = su.read_config_file(conffile)
        
        # Create project
        name = gParameters['name'] +'_rank' + str(rank)
        path_in = gParameters['path_in']
        path_out = gParameters['path_out']
        name_out = gParameters['name_out']
        global gpx
        gpx = G2sc.G2Project(newgpx=path_out+name+'.gpx')
    
        # Add phase: Requires CIF file
        cif = path_in + gParameters['cif']
        global phase
        phase = gpx.add_phase(cif,phasename=name,fmthint='CIF')
    
        # Get instrument file specification
        instprm = path_in + gParameters['instprm']
        # Histogram range
        Tmin = gParameters['tmin']
        Tmax = gParameters['tmax']
        Tstep = gParameters['tstep']
        hist = gpx.add_simulated_powder_histogram(name+'TOFsimulation',instprm,Tmin,Tmax,Tstep,phases=gpx.phases())
        hist.SampleParameters['Scale'][0] = 1000.
        # Set to no-background
        hist['Background'][0][3]=0.0
    
        symmetry = gParameters['symmetry']
    
        # Get ranges for sweeping parameters
        rangeDict = su.read_sweep_ranges(gParameters)
    
        # All symmetries sweep over a
        # Get start, stop and step
        tuple_a = rangeDict['cell_1']
        # Generate array (evenly spaced values within the given interval)
        a_range = np.arange(tuple_a[0], tuple_a[1], tuple_a[2])
        if rank == 0:
            print('tuple_a: ', tuple_a)
    
        # Configure sweep according to symmetry
        if symmetry == 'cubic':
            # Configure 1D sweeping grid
            grid_ = (a_range,)
            # Set sweeping function for symmetry given
            sweepf_ = cubic_lattice
        elif symmetry == 'trigonal':
            # Get start, stop and step
            tuple_alpha = rangeDict['cell_4']
            # Generate array (evenly spaced values within the given interval)
            alpha_range = np.arange(tuple_alpha[0], tuple_alpha[1], tuple_alpha[2])
            # Configure 2D sweeping grid
            grid_ = (a_range, alpha_range)
            # Set sweeping function for symmetry given
            sweepf_ = trigonal_lattice
            if rank == 0:
                print('tuple_alpha: ', tuple_alpha)
        elif symmetry == 'tetragonal':
            # Get start, stop and step
            tuple_c = rangeDict['cell_3']
            # Generate array (evenly spaced values within the given interval)
            c_range = np.arange(tuple_c[0], tuple_c[1], tuple_c[2])
            # Configure 2D sweeping grid
            grid_ = (a_range, c_range)
            # Set sweeping function for symmetry given
            sweepf_ = tetragonal_lattice
            if rank == 0:
                print('tuple_c: ', tuple_c)
        elif symmetry == 'trigonalHexP':
            # Get start, stop and step
            tuple_c = rangeDict['cell_3']
            # Generate array (evenly spaced values within the given interval)
            c_range = np.arange(tuple_c[0], tuple_c[1], tuple_c[2])
            # Configure 2D sweeping grid
            grid_ = (a_range, c_range)
            # Set sweeping function for symmetry given
            sweepf_ = trigonalHexP_lattice
            if rank == 0:
                print('tuple_c: ', tuple_c)
        elif symmetry == 'orthorhombic':
            # Get start, stop and step
            tuple_b = rangeDict['cell_2']
            tuple_c = rangeDict['cell_3']
            # Generate array (evenly spaced values within the given interval)
            b_range = np.arange(tuple_b[0], tuple_b[1], tuple_b[2])
            c_range = np.arange(tuple_c[0], tuple_c[1], tuple_c[2])
            # Configure 3D sweeping grid
            grid_ = (a_range, b_range, c_range)
            # Set sweeping function for symmetry given
            sweepf_ = orthorhombic_lattice
            if rank == 0:
                print('tuple_b: ', tuple_b)
                print('tuple_c: ', tuple_c)
        elif symmetry == 'monoclinic':
            # Get start, stop and step
            tuple_b = rangeDict['cell_2']
            tuple_c = rangeDict['cell_3']
            tuple_beta = rangeDict['cell_5']
            # Generate array (evenly spaced values within the given interval)
            b_range = np.arange(tuple_b[0], tuple_b[1], tuple_b[2])
            c_range = np.arange(tuple_c[0], tuple_c[1], tuple_c[2])
            beta_range = np.arange(tuple_beta[0], tuple_beta[1], tuple_beta[2])
            # Configure 4D sweeping grid
            grid_ = (a_range, b_range, c_range, beta_range)
            # Set sweeping function for symmetry given
            sweepf_ = monoclinic_lattice
            if rank == 0:
                print('tuple_b: ', tuple_b)
                print('tuple_c: ', tuple_c)
                print('tuple_beta: ', tuple_beta)
        else: # 'triclinic' by default
            # Get start, stop and step
            tuple_b = rangeDict['cell_2']
            tuple_c = rangeDict['cell_3']
            tuple_alpha = rangeDict['cell_4']
            tuple_beta = rangeDict['cell_5']
            tuple_gamma = rangeDict['cell_6']
            # Generate array (evenly spaced values within the given interval)
            b_range = np.arange(tuple_b[0], tuple_b[1], tuple_b[2])
            c_range = np.arange(tuple_c[0], tuple_c[1], tuple_c[2])
            alpha_range = np.arange(tuple_alpha[0], tuple_alpha[1], tuple_alpha[2])
            beta_range = np.arange(tuple_beta[0], tuple_beta[1], tuple_beta[2])
            gamma_range = np.arange(tuple_gamma[0], tuple_gamma[1], tuple_gamma[2])
            # Configure 6D sweeping grid
            grid_ = (a_range, b_range, c_range, alpha_range, beta_range, gamma_range)
            # Set sweeping function for symmetry given
            sweepf_ = triclinic_lattice
            if rank == 0:
                print('tuple_b: ', tuple_b)
                print('tuple_c: ', tuple_c)
                print('tuple_alpha: ', tuple_alpha)
                print('tuple_beta: ', tuple_beta)
                print('tuple_gamma: ', tuple_gamma)
    
        # Distribute computation
        nsim, histosz = su.grid_sweep(sweepf_, grid_, path_out, name_out + '_' + symmetry, comm)
        if rank == 0:
            print('----------------------------------------------------------')
            print('Number of simulations (%s): %d, size of histogram: %d' % (symmetry, nsim, histosz))
    
        end = time.time()
        print(i, "-th config, Rank = ", rank, " with Symmetry = ", symmetry, " with running time = ", end - start)


if __name__ == '__main__':
    main()

