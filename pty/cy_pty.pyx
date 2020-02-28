%%cython -a
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange
from cython cimport boundscheck, wraparound, cdivision

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t

@boundscheck(False)
@wraparound(False)
@cdivision(True)
def cy_squared(float[:,:,:] data4D_flat):
    cdef Py_ssize_t no_pos = data4D_flat.shape[0]
    cdef Py_ssize_t diff_size_q = data4D_flat.shape[1]
    cdef Py_ssize_t diff_size_p = data4D_flat.shape[2]
    
    cdef cnp.ndarray data4D_final = np.zeros([no_pos,diff_size_q,diff_size_p],dtype=np.float32)
    cdef float[:,:,:] data4D_final_view = data4D_final
    
    cdef cnp.ndarray temp = np.zeros([diff_size_q,diff_size_p],dtype=np.float32)
    cdef float[:,:] temp_view = temp
    cdef float data_min,data_max,data_range
    data_max = np.amax(data4D_flat)
    data_min = np.amin(data4D_flat)
    data_range = data_max - data_min
    cdef Py_ssize_t ii,jj,kk
    for ii in range(no_pos):
        temp_view = data4D_flat[ii,:,:]
        for jj in range(diff_size_q):
            for kk in prange(diff_size_p,nogil=True):
                temp_view[jj,kk] = (temp_view[jj,kk] - data_min)/data_range
                data4D_final_view[ii,jj,kk] = temp_view[jj,kk]**2
    return data4D_final

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void cy_resizer1D(float[:] data,float[:] res) nogil:   
    cdef Py_ssize_t M = data.shape[0]
    cdef Py_ssize_t N = res.shape[0]
    cdef float carry, data_sum, Mf, Nf, nf, mf, carry_mult, Rf, Rf_inv
    Mf = M
    Nf = N
    Rf = Mf/Nf
    Rf_inv = Nf/Mf
    cdef int n, m
    carry = 0
    m = 0
    for n in range(N):
        data_sum = carry
        nf = n
        while (((m*Nf) - (n*Mf)) < Mf):
            data_sum += data[m]
            m += 1
        mf = m
        carry_mult = (mf-((nf+1)*Rf))
        carry = carry_mult*data[m-1]
        data_sum = data_sum - carry
        data_sum = data_sum * Rf_inv
        res[n] = data_sum

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef cy_resizer2D(float[:,:] data, float sampling):
    cdef Py_ssize_t data_shape_y = data.shape[0]
    cdef Py_ssize_t data_shape_x = data.shape[1]
    cdef int sampled_shape_y, sampled_shape_x, yy, xx
    sampled_shape_y = np.round(data_shape_y/sampling)
    sampled_shape_x = np.round(data_shape_x/sampling)
    
    cdef cnp.ndarray resampled_x = np.zeros([data_shape_y,sampled_shape_x],dtype=np.float32)
    cdef float[:,:] resampled_x_view = resampled_x
    cdef cnp.ndarray resampled_f = np.zeros([sampled_shape_y,sampled_shape_x],dtype=np.float32)
    cdef float[:,:] resampled_f_view = resampled_f
    for yy in prange(data_shape_y,nogil=True):
        cy_resizer1D(data[yy,:],resampled_x_view[yy,:])
    for xx in prange(sampled_shape_x,nogil=True):
        cy_resizer1D(resampled_x_view[:,xx],resampled_f_view[:,xx])
    return resampled_f

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void cy_resizer2D_cyint(float[:,:] data_ini, float[:,:] data_s1, float[:,:] data_fin) nogil:
    cdef Py_ssize_t data_shape_y = data_ini.shape[0]
    cdef Py_ssize_t fina_shape_x = data_fin.shape[1]
    cdef int yy, xx
    for yy in range(data_shape_y):
        cy_resizer1D(data_ini[yy,:],data_s1[yy,:])
    for xx in range(fina_shape_x):
        cy_resizer1D(data_s1[:,xx],data_fin[:,xx])
        
@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef cy_resizer4Df(float[:,:,:] data4D_flat,float sampling):
    cdef Py_ssize_t no_pos = data4D_flat.shape[0]
    cdef Py_ssize_t qq_pos = data4D_flat.shape[1]
    cdef Py_ssize_t pp_pos = data4D_flat.shape[2]
    
    cdef int qq_res,pp_res,zz
    qq_res = np.round(qq_pos/sampling)
    pp_res = np.round(pp_pos/sampling)
    
    cdef cnp.ndarray resx_cbed = np.zeros([qq_pos,pp_res],dtype=np.float32)
    cdef float[:,:] resx_cbed_view = resx_cbed
    cdef cnp.ndarray res4D_flat = np.zeros([no_pos,qq_res,pp_res],dtype=np.float32)
    cdef float[:,:,:] res4D_flat_view = res4D_flat
    
    for zz in prange(no_pos,nogil=True):
        cy_resizer2D_cyint(data4D_flat[zz,:,:],resx_cbed_view,res4D_flat_view[zz,:,:])
    return res4D_flat