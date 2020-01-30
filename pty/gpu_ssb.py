import numpy as np
import scipy as sp
import warnings
from scipy import ndimage as scnd
import math
from scipy import optimize as sio
import numexpr as ne
import cupy as cp
import cupyx.scipy.ndimage as csnd
import numba

def get_flat_dpc(data4D_flat,
                 chunks=8,
                 centered=True):
    stops = np.zeros(chunks+1,dtype=np.int)
    stops[0:chunks] = np.arange(0,data4D_flat.shape[0],(data4D_flat.shape[0]/chunks))
    stops[chunks] = data4D_flat.shape[0]
    if centered:
        cent_x = cp.asarray(data4D_flat.shape[2])/2
        cent_y = cp.asarray(data4D_flat.shape[1])/2
    else:
        CentralDisk = np.median(data4D_flat,axis=0)
        cent_x,cent_y,_ = st.util.sobel_circle(CentralDisk)
        cent_x = cp.asarray(cent_x)
        cent_y = cp.asarray(cent_y)
    yy, xx = cp.mgrid[0:data4D_flat.shape[1],0:data4D_flat.shape[2]]
    FlatSum = cp.asarray(np.sum(data4D_flat,axis=(-1,-2)))
    YCom_CPU = np.zeros(data4D_flat.shape[0],dtype=data4D_flat.dtype)
    XCom_CPU = np.zeros(data4D_flat.shape[0],dtype=data4D_flat.dtype)
    for ii in range(chunks):
        startval = stops[ii]
        stop_val = stops[ii+1]
        gpu_4Dchunk = cp.asarray(data4D_flat[startval:stop_val,:,:])
        FlatY = cp.multiply(gpu_4Dchunk,yy)
        FlatX = cp.multiply(gpu_4Dchunk,xx)
        YCom = (cp.sum(FlatY,axis=(-1,-2))/FlatSum[startval:stop_val]) - cent_y
        XCom = (cp.sum(FlatX,axis=(-1,-2))/FlatSum[startval:stop_val]) - cent_x
        YCom_CPU[startval:stop_val] = cp.asnumpy(YCom)
        XCom_CPU[startval:stop_val] = cp.asnumpy(XCom)
    del YCom, XCom, gpu_4Dchunk, cent_x, cent_y, FlatSum
    return YCom_CPU,XCom_CPU

def cart2pol(x, y):
    rho = ne.evaluate("((x**2) + (y**2)) ** 0.5")
    phi = ne.evaluate("arctan2(y, x)")
    return (rho, phi)

def pol2cart(rho, phi):
    x = ne.evaluate("rho * cos(phi)")
    y = ne.evaluate("rho * sin(phi)")
    return (x, y)

def angle_fun(angle,rho_dpc,phi_dpc):
    x_dpc,y_dpc = pol2cart(rho_dpc,(phi_dpc + (angle*((np.pi)/180))))
    charge = np.gradient(x_dpc)[1] + np.gradient(y_dpc)[0]
    angle_sum = np.sum(np.abs(charge))
    return angle_sum

def optimize_angle(x_dpc,y_dpc,adf_stem):
    chg_sums = np.zeros(2,dtype=x_dpc.dtype)
    angles = np.zeros(2,dtype=x_dpc.dtype)
    x0 = 90
    rho_dpc,phi_dpc = cart2pol(x_dpc,y_dpc)
    x = sio.minimize(angle_fun,x0,args=(rho_dpc,phi_dpc))
    min_x = x.x
    sol1 = min_x - 90
    sol2 = min_x + 90
    chg_sums[0] = np.sum(charge_dpc(x_dpc,y_dpc,sol1)*adf_stem)
    chg_sums[1] = np.sum(charge_dpc(x_dpc,y_dpc,sol2)*adf_stem)
    angles[0] = sol1
    angles[1] = sol2
    angle = angles[chg_sums==np.amin(chg_sums)][0]
    return angle

def corrected_dpc(x_dpc,y_dpc,angle):
    rho_dpc,phi_dpc = cart2pol(x_dpc,y_dpc)
    x_dpc2,y_dpc2 = pol2cart(rho_dpc,(phi_dpc + (angle*((np.pi)/180))))
    return x_dpc2,y_dpc2

def potential_dpc(x_dpc,y_dpc,angle=0):
    if angle==0:
        potential = integrate_dpc(x_dpc,y_dpc)
    else:
        rho_dpc,phi_dpc = cart2pol(x_dpc,y_dpc)
        x_dpc,y_dpc = pol2cart(rho_dpc,phi_dpc + (angle*((np.pi)/180)))
        potential = integrate_dpc(x_dpc,y_dpc)
    return potential

def charge_dpc(x_dpc,y_dpc,angle=0):
    if angle==0:
        charge = np.gradient(x_dpc)[1] + np.gradient(y_dpc)[0]
    else:
        rho_dpc,phi_dpc = cart2pol(x_dpc,y_dpc)
        x_dpc,y_dpc = pol2cart(rho_dpc,phi_dpc + (angle*((np.pi)/180)))
        charge = np.gradient(x_dpc)[1] + np.gradient(y_dpc)[0]
    return charge

def integrate_dpc(xshift,
                  yshift,
                  fourier_calibration=1):
    #Initialize matrices
    size_array = np.asarray(np.shape(xshift))
    x_mirrored = np.zeros(2*size_array,dtype=np.float64)
    y_mirrored = np.zeros(2*size_array,dtype=np.float64)
    
    #Generate antisymmetric X arrays
    x_mirrored[0:size_array[0],0:size_array[1]] = np.fliplr(np.flipud(0 - xshift))
    x_mirrored[0:size_array[0],size_array[1]:(2*size_array[1])] = np.fliplr(0 - xshift)
    x_mirrored[size_array[0]:(2*size_array[0]),0:size_array[1]] = np.flipud(xshift)
    x_mirrored[size_array[0]:(2*size_array[0]),size_array[1]:(2*size_array[1])] = xshift
    
    #Generate antisymmetric Y arrays
    y_mirrored[0:size_array[0],0:size_array[1]] = np.fliplr(np.flipud(0 - yshift))
    y_mirrored[0:size_array[0],size_array[1]:(2*size_array[1])] = np.fliplr(yshift)
    y_mirrored[size_array[0]:(2*size_array[0]),0:size_array[1]] = np.flipud(0 - yshift)
    y_mirrored[size_array[0]:(2*size_array[0]),size_array[1]:(2*size_array[1])] = yshift
    
    #Calculated Fourier transform of antisymmetric matrices
    x_mirr_ft = np.fft.fft2(x_mirrored)
    y_mirr_ft = np.fft.fft2(y_mirrored)
    
    #Calculated inverse Fourier space calibration
    qx = np.mean(np.diff((np.arange(-size_array[1],size_array[1], 1))/
                         (2*fourier_calibration*size_array[1])))
    qy = np.mean(np.diff((np.arange(-size_array[0],size_array[0], 1))/
                         (2*fourier_calibration*size_array[0])))
    
    #Calculate mirrored CPM integrand
    mirr_ft = (x_mirr_ft + ((1j)*y_mirr_ft))/(qx + ((1j)*qy))
    mirr_int = np.fft.ifft2(mirr_ft)
    
    #Select integrand from antisymmetric matrix
    integrand = np.abs(mirr_int[size_array[0]:(2*size_array[0]),size_array[1]:(2*size_array[1])])
    
    return integrand

def centerCBED(data4D_flat,
               x_cen,
               y_cen,
               chunks=8):
    stops = np.zeros(chunks+1,dtype=np.int)
    stops[0:chunks] = np.arange(0,data4D_flat.shape[0],(data4D_flat.shape[0]/chunks))
    stops[chunks] = data4D_flat.shape[0]
    max_size = int(np.amax(np.diff(stops)))
    centered4D = np.zeros_like(data4D_flat)
    image_size = np.asarray(data4D_flat.shape[1:3])
    fourier_cal_y = (cp.linspace((-image_size[0]/2), ((image_size[0]/2) - 1), image_size[0]))/image_size[0]
    fourier_cal_x = (cp.linspace((-image_size[1]/2), ((image_size[1]/2) - 1), image_size[1]))/image_size[1]
    [fourier_mesh_x, fourier_mesh_y] = cp.meshgrid(fourier_cal_x, fourier_cal_y)
    move_pixels = np.flip(image_size/2) - np.asarray((x_cen,y_cen))
    move_phase = cp.exp((-2) * np.pi * 1j * ((fourier_mesh_x*move_pixels[0]) + (fourier_mesh_y*move_pixels[1])))
    for ii in range(chunks):
        startval = stops[ii]
        stop_val = stops[ii+1]
        gpu_4Dchunk = cp.asarray(data4D_flat[startval:stop_val,:,:])
        FFT_4D = cp.fft.fftshift(cp.fft.fft2(gpu_4Dchunk,axes=(-1,-2)),axes=(-1,-2))
        FFT_4Dmove = cp.absolute(cp.fft.ifft2(cp.multiply(FFT_4D,move_phase),axes=(-1,-2)))
        centered4D[startval:stop_val,:,:] = cp.asnumpy(FFT_4Dmove)
    del FFT_4D,gpu_4Dchunk,FFT_4Dmove,move_phase,fourier_cal_y,fourier_cal_x,fourier_mesh_x,fourier_mesh_y
    return centered4D

def wavelength_pm(voltage_kV):
    m = 9.109383 * (10 ** (-31))  # mass of an electron
    e = 1.602177 * (10 ** (-19))  # charge of an electron
    c = 299792458  # speed of light
    h = 6.62607 * (10 ** (-34))  # Planck's constant
    voltage = voltage_kV * 1000
    numerator = (h ** 2) * (c ** 2)
    denominator = (e * voltage) * ((2*m*(c ** 2)) + (e * voltage))
    wavelength_pm = (10 ** 12) *((numerator/denominator) ** 0.5) #in picometers
    return wavelength_pm

def get_sampling(datashape,aperture_mrad,voltage,calibration_pm,radius_pixels):
    yscanf = (np.linspace((-datashape[0]/2), 
                          ((datashape[0]/2) - 1), datashape[0]))/(calibration_pm*datashape[0])
    xscanf = (np.linspace((-datashape[1]/2), 
                          ((datashape[1]/2) - 1), datashape[1]))/(calibration_pm*datashape[1])
    [xscanf_m, yscanf_m] = np.meshgrid(xscanf, yscanf)
    scanf_m = 1000*wavelength_pm(voltage)*(((xscanf_m**2) + (yscanf_m)**2)**0.5)
    fourier_beam = np.zeros_like(scanf_m)
    fourier_beam[scanf_m < aperture_mrad] = 1
    real_rad = (np.sum(fourier_beam)/np.pi)**0.5
    sampling = radius_pixels/real_rad
    return sampling

@numba.jit
def resizer1D(data,N):   
    M = data.size
    res = np.zeros(N,dtype=data.dtype)
    carry=0
    m=0
    for n in range(int(N)):
        data_sum = carry
        while m*N - n*M < M :
            data_sum += data[m]
            m += 1
        carry = (m-(n+1)*M/N)*data[m-1]
        data_sum -= carry
        res[n] = data_sum*N/M
    return res

@numba.jit
def resizer1D_numbaopt(data,res,N):   
    M = data.size
    carry=0
    m=0
    for n in range(int(N)):
        data_sum = carry
        while m*N - n*M < M :
            data_sum += data[m]
            m += 1
        carry = (m-(n+1)*M/N)*data[m-1]
        data_sum -= carry
        res[n] = data_sum*N/M
    return res

@numba.jit
def resizer2D(data2D,sampling):
    data_shape = np.asarray(np.shape(data2D))
    sampled_shape = (np.round(data_shape/sampling)).astype(int)
    resampled_x = np.zeros((data_shape[0],sampled_shape[1]),dtype=data2D.dtype)
    resampled_f = np.zeros(sampled_shape,dtype=data2D.dtype)
    for yy in range(data_shape[0]):
        resampled_x[yy,:] = resizer1D_numbaopt(data2D[yy,:],resampled_x[yy,:],sampled_shape[1])
    for xx in range(sampled_shape[1]):
        resampled_f[:,xx] = resizer1D_numbaopt(resampled_x[:,xx],resampled_f[:,xx],sampled_shape[0])
    return resampled_f

@numba.jit
def resizer2D_numbaopt(data2D,resampled_x,resampled_f,sampling):
    data_shape = np.asarray(np.shape(data2D))
    sampled_shape = (np.round(data_shape/sampling)).astype(int)
    for yy in range(data_shape[0]):
        resampled_x[yy,:] = resizer1D_numbaopt(data2D[yy,:],resampled_x[yy,:],sampled_shape[1])
    for xx in range(sampled_shape[1]):
        resampled_f[:,xx] = resizer1D_numbaopt(resampled_x[:,xx],resampled_f[:,xx],sampled_shape[0])
    return resampled_f

@numba.jit
def resizer4Df(data4D_flat,sampling):
    datashape = np.asarray(data4D_flat.shape)
    res_shape = np.copy(datashape)
    res_shape[1:3] = np.round(datashape[1:3]/sampling)
    data4D_res = np.zeros(res_shape.astype(int),dtype=data4D_flat.dtype)
    resampled_x = np.zeros((datashape[1],res_shape[2]),data4D_flat.dtype)
    resampled_f = np.zeros(res_shape[1:3],dtype=data4D_flat.dtype)
    for zz in range(data4D_flat.shape[0]):
        data4D_res[zz,:,:] = resizer2D_numbaopt(data4D_flat[zz,:,:],resampled_x,resampled_f,sampling)
    return data4D_res

@numba.jit
def resizer4D(data4D,sampling):
    data4D_flat = np.reshape(data4D,(data4D.shape[0]*data4D.shape[1],data4D.shape[2],data4D.shape[3]))
    datashape = np.asarray(data4D_flat.shape)
    res_shape = np.copy(datashape)
    res_shape[1:3] = np.round(datashape[1:3]/sampling)
    data4D_res = np.zeros(res_shape.astype(int),dtype=data4D_flat.dtype)
    resampled_x = np.zeros((datashape[1],res_shape[2]),data4D_flat.dtype)
    resampled_f = np.zeros(res_shape[1:3],dtype=data4D_flat.dtype)
    for zz in range(data4D_flat.shape[0]):
        data4D_res[zz,:,:] = resizer2D_numbaopt(data4D_flat[zz,:,:],resampled_x,resampled_f,sampling)
    res_4D = np.reshape(data4D_res,(data4D.shape[0],data4D.shape[1],resampled_f.shape[0],resampled_f.shape[1]))
    return res_4D

def subpixel_pad2D(initial_array,final_size):
    final_size = np.asarray(final_size)
    padded = np.amin(initial_array)*(np.ones(final_size,dtype=initial_array.dtype))
    padded[0:initial_array.shape[0],0:initial_array.shape[1]] = initial_array
    fourier_cal_y = (np.linspace((-final_size[0]/2), ((final_size[0]/2) - 1), final_size[0]))/final_size[0]
    fourier_cal_x = (np.linspace((-final_size[1]/2), ((final_size[1]/2) - 1), final_size[1]))/final_size[1]
    [fourier_mesh_x, fourier_mesh_y] = np.meshgrid(fourier_cal_x, fourier_cal_y)
    move_pixels = np.flip(0.5*(final_size - np.asarray(initial_array.shape)))
    move_phase = np.exp((-2) * np.pi * 1j * ((fourier_mesh_x*move_pixels[0]) + (fourier_mesh_y*move_pixels[1])))
    padded_f = np.fft.fftshift(np.fft.fft2(padded))
    padded_c = np.abs(np.fft.ifft2(np.multiply(padded_f,move_phase))) 
    return padded_c

def subpixel_pad4D(data4D_flat,final_size,cut_radius,chunks=10):
    stops = np.zeros(chunks+1,dtype=np.int)
    stops[0:chunks] = np.arange(0,data4D_flat.shape[0],(data4D_flat.shape[0]/chunks))
    stops[chunks] = data4D_flat.shape[0]
    max_size = int(np.amax(np.diff(stops)))
    
    final_size = (np.asarray(final_size)).astype(int)
    move_pixels = cp.asarray(np.flip(0.5*(final_size - np.asarray(data4D_flat.shape[1:3]))))
    
    yy,xx = np.mgrid[0:final_size[0],0:final_size[1]]
    rad = ((yy - final_size[0]/2)**2) + ((xx - final_size[1]/2)**2)
    cutoff = cp.asarray((rad < ((1.1*cut_radius)**2)).astype(data4D_flat.dtype))
    
    cbed = cp.zeros(final_size,dtype=data4D_flat.dtype)
    
    fourier_cal_y = (cp.linspace((-final_size[0]/2), ((final_size[0]/2) - 1), final_size[0]))/final_size[0]
    fourier_cal_x = (cp.linspace((-final_size[1]/2), ((final_size[1]/2) - 1), final_size[1]))/final_size[1]
    [fourier_mesh_x, fourier_mesh_y] = cp.meshgrid(fourier_cal_x, fourier_cal_y)
    move_phase = cp.exp((-2) * np.pi * (1j) * ((fourier_mesh_x*move_pixels[0]) + (fourier_mesh_y*move_pixels[1])))
    
    padded_4D = np.zeros((data4D_flat.shape[0],final_size[0],final_size[1]),dtype=data4D_flat.dtype)
    padded_on_gpu = cp.zeros((max_size,final_size[0],final_size[1]),dtype=data4D_flat.dtype)
    for cc in range(chunks):
        startval = stops[cc]
        stop_val = stops[cc+1]
        gpu_4Dchunk = cp.asarray(data4D_flat[startval:stop_val,:,:])
        for ii in range(gpu_4Dchunk.shape[0]):
            cbed[0:data4D_flat.shape[1],0:data4D_flat.shape[2]] = gpu_4Dchunk[ii,:,:]
            FFT_cbd = cp.fft.fftshift(cp.fft.fft2(cbed))
            moved_cbed = (cp.absolute(cp.fft.ifft2(cp.multiply(FFT_cbd,move_phase)))).astype(data4D_flat.dtype)
            padded_on_gpu[ii,:,:] = moved_cbed*cutoff
        padded_4D[startval:stop_val,:,:] = cp.asnumpy(padded_on_gpu[0:gpu_4Dchunk.shape[0],:,:])
    del padded_on_gpu, moved_cbed, cbed, FFT_cbd, move_phase, gpu_4Dchunk, move_pixels, cutoff
    return padded_4D

def gpu_rotator(data4D_flat,rotangle,axes,chunks=40):
    stops = np.zeros(chunks+1,dtype=np.int)
    stops[0:chunks] = np.arange(0,data4D_flat.shape[0],(data4D_flat.shape[0]/chunks))
    stops[chunks] = data4D_flat.shape[0]
    max_size = int(np.amax(np.diff(stops)))
    data4D_rot = np.zeros_like(data4D_flat)
    for cc in range(chunks):
        startval = stops[cc]
        stop_val = stops[cc+1]
        gpu_4Dchunk = cp.asarray(data4D_flat[startval:stop_val,:,:])
        data4D_rot[startval:stop_val,:,:] = cp.asnumpy(csnd.rotate(gpu_4Dchunk,rotangle,axes,reshape=False))
    del gpu_4Dchunk
    return data4D_rot

def get_G_matrix(data4D,chunks=20):
    data4D = np.transpose(data4D,(2,3,0,1)) #real in 2,3
    data_shape = data4D.shape
    data4D = np.reshape(data4D,(data_shape[0]*data_shape[1],data_shape[2],data_shape[3]))
    stops = np.zeros(chunks+1,dtype=np.int)
    stops[0:chunks] = np.arange(0,data4D.shape[0],(data4D.shape[0]/chunks))
    stops[chunks] = data4D.shape[0]
    max_size = int(np.amax(np.diff(stops)))
    data4DF = np.zeros_like(data4D,dtype=np.complex64)
    for cc in range(chunks):
        startval = stops[cc]
        stop_val = stops[cc+1]
        gpu_4Dchunk = cp.asarray(data4D[startval:stop_val,:,:])
        gpu_4DF = cp.fft.fftshift(cp.fft.fft2(gpu_4Dchunk,axes=(1,2)),axes=(1,2)) #now real is Q' which is 2,3
        data4DF[startval:stop_val,:,:] = cp.asnumpy(gpu_4DF)
    del gpu_4Dchunk, gpu_4DF
    data4DF = np.reshape(data4DF,data_shape)
    return data4DF

def lobe_calc(data4DF,Four_Y,Four_X,FourXY,rsize,cutoff,chunks):
    stops = np.zeros(chunks+1,dtype=np.int)
    stops[0:chunks] = np.arange(0,data4DF.shape[-1],(data4DF.shape[-1]/chunks))
    stops[chunks] = data4DF.shape[-1]
    
    left_image = cp.zeros_like(FourXY,dtype=np.complex64)
    rightimage = cp.zeros_like(FourXY,dtype=np.complex64)
    d_zero = FourXY < cutoff
    
    for cc in range(chunks):
        startval = stops[cc]
        stop_val = stops[cc+1]
        gpu_4Dchunk = cp.asarray(data4DF[:,:,startval:stop_val])
        rcalc = rsize[startval:stop_val,:]
        for pp in range(rcalc.shape[0]):
            ii,jj = rcalc[pp,:]
            xq = Four_X[ii,jj]
            yq = Four_Y[ii,jj]

            cbd = gpu_4Dchunk[:,:,pp]
            cbd_phase = cp.angle(cbd)
            cbd_ampli = cp.absolute(cbd)

            d_plus = (((Four_X + xq)**2) + ((Four_Y + yq)**2))**0.5
            d_minu = (((Four_X - xq)**2) + ((Four_Y - yq)**2))**0.5

            ll = cp.logical_and((d_plus < cutoff),(d_minu > cutoff))
            ll = cp.logical_and(ll,d_zero)

            rr = cp.logical_and((d_plus > cutoff),(d_minu < cutoff))
            rr = cp.logical_and(rr,d_zero)

            left_trotter = cp.multiply(cbd_ampli[ll],cp.exp((1j)*cbd_phase[ll]))
            righttrotter = cp.multiply(cbd_ampli[rr],cp.exp((1j)*cbd_phase[rr]))

            left_image[ii,jj] = cp.sum(left_trotter)
            rightimage[ii,jj] = cp.sum(righttrotter)
    
    del gpu_4Dchunk,d_plus,d_minu,ll,rr,left_trotter,righttrotter,cbd,cbd_phase,cbd_ampli,d_zero, rcalc
    return left_image,rightimage
    
def ssb_kernel(processed4D,real_calibration,aperture,voltage,chunks=12):
    data_size = np.asarray(processed4D.shape)
    processed4D = np.reshape(processed4D,(data_size[0],data_size[1],data_size[2]*data_size[3]))
    wavelength = wavelength_pm(voltage)
    cutoff = aperture/(1000*wavelength)
    four_y = cp.fft.fftshift(cp.fft.fftfreq(data_size[0], real_calibration))
    four_x = cp.fft.fftshift(cp.fft.fftfreq(data_size[1], real_calibration))
    Four_X,Four_Y = cp.meshgrid(four_x,four_y)
    FourXY = cp.sqrt((Four_Y ** 2) + (Four_X**2))
    yy,xx = cp.mgrid[0:data_size[0],0:data_size[1]]
    rsize = cp.zeros((np.size(yy),2),dtype=int)
    rsize[:,0] = cp.ravel(yy)
    rsize[:,1] = cp.ravel(xx)
    
    left_imGPU,rightimGPU = lobe_calc(processed4D,Four_Y,Four_X,FourXY,rsize,cutoff,chunks)
    left_image = cp.asnumpy(cp.fft.ifft2(left_imGPU))
    rightimage = cp.asnumpy(cp.fft.ifft2(rightimGPU))
    
    del four_y, four_x, Four_X, Four_Y, FourXY, yy, xx, rsize, left_imGPU, rightimGPU
    
    return left_image,rightimage