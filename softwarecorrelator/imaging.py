import numpy
from  astropy.coordinates.name_resolve import get_icrs_coordinates
import astropy.coordinates as acc
import softwarecorrelator.coordinates as scc


def antsol(vis_data_vect, vis_model_vect,
                gain_solutions, antenna1_vect, antenna2_vect,
                baseline_weights, alpha=0.5, epsilon=1e-5, refant=0,
                max_iter=200):
    vis_mask = vis_model_vect == 0.0
    model = ma.array(vis_model_vect, mask=vis_mask)
    vis_pointsource = ma.array(vis_data_vect/model, mask=vis_mask)
    num_ant = len(gain_solutions)
    w = baseline_weights
    x_mat = numpy.zeros((num_ant, num_ant ), dtype=numpy.complex64)
    w_mat = numpy.zeros((num_ant, num_ant ), dtype=numpy.float32)
    ant1 = antenna1_vect
    ant2 = antenna2_vect
    for a1, a2, v, w in zip(ant1, ant2, vis_pointsource, w):
        x_mat[a1, a2] = v
        x_mat[a2, a1] = numpy.conj(v)
        w_mat[a1, a2] = w
        w_mat[a2, a1] = w
        if a1 == a2:
            w_mat[a1, a2] = 0

    g_prev = gain_solutions
    g_next = numpy.zeros((len(g_prev),), dtype=numpy.complex64)
    iteration=0
    mask = False
    while True:
        den = ma.array((w_mat*(numpy.abs(g_prev)**2)[numpy.newaxis, :]).sum(axis=1), mask=mask)
        mask = den == 0.0
        den.mask = mask
        if mask.all():
            break
        num = ma.array((x_mat*w_mat*g_prev[numpy.newaxis,:]).sum(axis=1), mask=mask)
        g_next = g_prev + alpha*(num/den - g_prev)
        if norm(g_next-g_prev)/norm(g_next) < epsilon:
            break
        iteration += 1
        g_prev = g_next
        if iteration >= max_iter:
            mask = mask+True
            break
    return ma.array(g_next/exp(1.j*angle(g_next[refant])), mask=mask)



def baseline_matrix_m(positions_m):
    r'''
    Computes all baseline pairs between the provided positions.

    **Parameters**

    positions_m : (n, 3) numpy.array of floats
        The antenna positions in meters.

    **Returns**

    An (n, n, 3) numpy.array of floats containing the matrix of
    baseline vectors. baseline_matrix_m[i, j] contains positions_m[j]
    - positions_m[i].

    **Examples**

    >>> baseline_matrix_m(numpy.array([[1, 2, 3.0], [-2, -1, 0], [3, 4, 2]]))
    array([[[ 0.,  0.,  0.],
            [-3., -3., -3.],
            [ 2.,  2., -1.]],
    <BLANKLINE>
           [[ 3.,  3.,  3.],
            [ 0.,  0.,  0.],
            [ 5.,  5.,  2.]],
    <BLANKLINE>
           [[-2., -2.,  1.],
            [-5., -5., -2.],
            [ 0.,  0.,  0.]]])
    '''
    return positions_m[numpy.newaxis, :, :] - positions_m[:, numpy.newaxis,  :]


def acm_as_vector(acm, include_diag=False):
    r'''
    Convert array correlation matrix to a vector of the upper
    triangular matrix, excluding the auto correlations.

    **Parameters**

    acm : (n, n) numpy matrix
        The matrix to unravel.

    include_diag : bool
        If True, include the on-diagonal elements.

    **Returns**

    A numpy.array containing the n*(n-1)/2 upper triangular elements
    of the acm, where consecutive rows are appended.

    **Examples**

    >>> acm_as_vector(numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    array([2, 3, 6])
    >>> acm_as_vector(numpy.array([[[1, 1], [2, 2], [3, 3]],
    ...                      [[4, 4], [5, 5], [6, 6]],
    ...                      [[7, 7], [8, 8], [9, 9]]]))
    array([[2, 2],
           [3, 3],
           [6, 6]])

    >>> acm_as_vector(numpy.array([[[1, 1], [2, 2], [3, 3]],
    ...                      [[4, 4], [5, 5], [6, 6]],
    ...                      [[7, 7], [8, 8], [9, 9]]]),
    ...               include_diag=True)
    array([[1, 1],
           [2, 2],
           [3, 3],
           [5, 5],
           [6, 6],
           [9, 9]])
    '''
    upper_triangle_vector = []
    acm_list              = acm.tolist()
    offset = 1
    if include_diag:
        offset = 0
    for row_num, row in enumerate(acm_list):
        upper_triangle_vector += row[row_num+offset:]
    return numpy.array(upper_triangle_vector, dtype=acm.dtype)




def make_imaging_matrix(uvw_m, freq_hz, l_rad, m_rad,
                        min_baseline_lambda=None,
                        max_baseline_lambda=None):
    r'''
    Compute a data model matrix for use in image reconstruction
    algorithms. It assumes the following data model:
    
    .. math:: \vec{v} = \mathrm{A}\vec{x},
    
    where :math:`\vec{x}` is the image in the x,y domain we are after, and
    :math:`\vec{v}` are the observed visibilities in the l,m
    domain. Because the matrix describes the transform from x,y to l,m,
    the sign in the exponent is the opposite of *fourier_sign*. The matrix
    elements are given by
    
    .. math:: a_{ij} = \mathrm{e}^{-s 2 \pi \mathrm{i} \nu (x_i l_j + y_i m_j)/c}.
    
    
    **Parameters**
    
    uvw_m : (n, 3) numpy.array of floats
        The uvw coordinates of the baselines in m

    freq_hz : float
        The observing frequency in Hz.

    l_rad : 1D numpy.array of floats
        l coordinates of the image pixels in radians.

    m_rad : 1D numpy.array of floats
        m coordinates of the image pixels in radians.

    min_baseline_lambda: float
        Minimum baseline to use in imaging. If None: use the shortest.

    max_baseline_lambda: float
        Maximum baseline to use in imaging. If None: use longest.

    **Returns**
    
    A 2D numpy array representing the matrix :math:`\mathrm{A}`, with
    dimensions (len(FTGrid.l_rad), len(FTGrid.x_m)*len(FTGrid.y_m)).
    
    **Examples**
    
    >>> l_rad = numpy.arange(-1,+1.001, 0.5)
    >>> m_rad = numpy.arange(-1,+1.001, 0.25)
    >>> freq_hz = 59e6
    >>> uvw_m = numpy.array([[1, 2, 3.0], [-2, -1, 0], [3, 4, 2]])
    >>> mat, grid_l, grid_m = make_imaging_matrix(uvw_m, freq_hz, l_rad, m_rad)
    >>> mat.shape
    (45, 3)
    >>> mat[0]
    array([-0.84294999+0.53799194j, -0.84294999-0.53799194j,
           -0.71864951-0.69537246j])

    '''
    arg    = 2j*numpy.pi*freq_hz/299792458.0

    u_m    = numpy.array(uvw_m[:, 0], dtype=numpy.float32)
    v_m    = numpy.array(uvw_m[:, 1], dtype=numpy.float32)
    w_m    = numpy.array(uvw_m[:, 2], dtype=numpy.float32)

    
    grid_l, grid_m = numpy.meshgrid(l_rad, m_rad)
    grid_l = numpy.array(grid_l.ravel(), dtype=numpy.float32)[:, numpy.newaxis]
    grid_m = numpy.array(grid_m.ravel(), dtype=numpy.float32)[:, numpy.newaxis]
    grid_n_squared = 1 - grid_l**2 - grid_m**2
    grid_n_squared[grid_n_squared < 0] = 0
    grid_n = numpy.sqrt(grid_n_squared)

    mat    = numpy.exp(arg*(u_m[numpy.newaxis, :]*grid_l \
                            + v_m[numpy.newaxis, :]*grid_m \
                            + w_m[numpy.newaxis, :]*grid_n))

    uv_lambda = numpy.sqrt(u_m**2 + v_m**2)*freq_hz/299792458.0
    bl_mask = numpy.ones(u_m.shape[0], dtype=numpy.int)
    if min_baseline_lambda is not None:
        bl_mask *= uv_lambda >= min_baseline_lambda
    if max_baseline_lambda is not None:
        bl_mask *= uv_lambda <= max_baseline_lambda
    
    return mat.reshape((len(l_rad)*len(m_rad), len(uvw_m)))*bl_mask[numpy.newaxis,:], grid_l, grid_m





def make_predict_matrix(uvw_m, freq_hz, source_lmn_rad,
                        min_baseline_lambda=None,
                        max_baseline_lambda=None):
    r'''
    Compute a data model matrix for use in image reconstruction
    algorithms. It assumes the following data model:
    
    .. math:: \vec{v} = \mathrm{A}\vec{x},
    
    where :math:`\vec{x}` is the image in the x,y domain we are after, and
    :math:`\vec{v}` are the observed visibilities in the l,m
    domain. Because the matrix describes the transform from x,y to l,m,
    the sign in the exponent is the opposite of *fourier_sign*. The matrix
    elements are given by
    
    .. math:: a_{ij} = \mathrm{e}^{-s 2 \pi \mathrm{i} \nu (x_i l_j + y_i m_j)/c}.
    
    
    **Parameters**
    
    uvw_m : (n, 3) numpy.array of floats
        The uvw coordinates of the baselines in m

    freq_hz : float
        The observing frequency in Hz.

    source_lmn_rad : 2D numpy.array of floats
        lmn coordinates of the sources to solve for in radians.

    min_baseline_lambda: float
        Minimum baseline to use in imaging. If None: use the shortest.

    max_baseline_lambda: float
        Maximum baseline to use in imaging. If None: use longest.

    **Returns**
    
    A 2D numpy array representing the matrix :math:`\mathrm{A}`, with
    dimensions (len(uvw_m), len(source_lmn_rad)).
    
    **Examples**
    
    >>> source_lmn_rad = numpy.array([[0.4,  0.5, 0.76811],
    ...                               [0.3, -0.2, 0.93274],
    ...                               [-0.5, 0.25, 0.82916]])
    >>> freq_hz = 59e6
    >>> uvw_m = numpy.array([[1, 2, 3.0], [-2, -1, 0], [3, 4, 2]])
    >>> mat = make_predict_matrix(uvw_m, freq_hz, source_lmn_rad)
    >>> mat.shape
    (3, 3)
    >>> mat[0]
    array([-0.13142364+0.99132627j, -0.98106945+0.19365615j,
           -0.99784231-0.06565581j])

    '''
    arg    = -2j*numpy.pi*freq_hz/299792458.0

    u_m    = numpy.array(uvw_m[:, 0], dtype=numpy.float32)
    v_m    = numpy.array(uvw_m[:, 1], dtype=numpy.float32)
    w_m    = numpy.array(uvw_m[:, 2], dtype=numpy.float32)

    
    source_l = numpy.array(source_lmn_rad[:,0], dtype=numpy.float32)[numpy.newaxis,:]
    source_m = numpy.array(source_lmn_rad[:,1], dtype=numpy.float32)[numpy.newaxis,:]
    source_n = numpy.array(source_lmn_rad[:,2], dtype=numpy.float32)[numpy.newaxis,:]

    mat    = numpy.exp(arg*(  u_m[:, numpy.newaxis]*source_l \
                            + v_m[:, numpy.newaxis]*source_m \
                            + w_m[:, numpy.newaxis]*source_n))

    uv_lambda = numpy.sqrt(u_m**2 + v_m**2)*freq_hz/299792458.0
    bl_mask = numpy.ones(u_m.shape[0], dtype=numpy.int)
    if min_baseline_lambda is not None:
        bl_mask *= uv_lambda >= min_baseline_lambda
    if max_baseline_lambda is not None:
        bl_mask *= uv_lambda <= max_baseline_lambda
    
    return mat*bl_mask[:, numpy.newaxis]







def near_field_distances(pqr_ant, pqr_pixels):
    r'''
    Calculate distance to pixels from antenna positions.

    **Parameters**

    pqr_ant : numpy.array of floats
        PQR coordinates of every antenna. Indices [ant, pqr]

    pqr_pixels : numpy.array of floats
        PQR coordinates of every pixel. Indices [pixel, pqr]


    **Returns**

    2D numpy.array of float distances [ant, pixel]


    **Examples**

    >>> nfd = near_field_distances(numpy.array([[1,2,3],[-2,-3,-1],[0,-2, +4]]),
    ...                            numpy.array([[10,5,6],[-4,-3,-2]]))
    >>> numpy.abs(nfd - numpy.array([[  9.94987437,   8.66025404], [ 16.03121954,   2.23606798], [ 12.36931688,   7.28010989]])).max() < 1e-6
    True
    '''
    return numpy.array(numpy.linalg.norm(pqr_ant[:, numpy.newaxis, :]
                                  - pqr_pixels[numpy.newaxis, :, :], axis=-1),
                       dtype=numpy.float32)



def predict_matrix_near_field(ant_pqr_m,
                              freq_hz,
                              source_pqr_m,
                              phase_only=True,
                              min_baseline_lambda=None,
                              max_baseline_lambda=None):
    r'''
    **Examples**

    >>> ant_pqr_m = numpy.array([[0,0,0], [10, 20, 0], [-5, -7, +2]])
    >>> freq_hz = 250e6
    >>> source_pqr_m = numpy.array([[0, 50, 10], [30, 0, -3]])
    >>> predict_matrix_near_field(ant_pqr_m, freq_hz, source_pqr_m)
    array([[ 1.00000000+0.j        ,  1.00000000+0.j        ],
           [ 0.65453351+0.756033j  , -0.88592178-0.46383473j],
           [-0.54493666-0.83847719j,  0.85561699-0.51760942j],
           [ 1.00000000+0.j        ,  1.00000000+0.j        ],
           [-0.99059576-0.13682133j, -0.51792449+0.85542631j],
           [ 0.99999994+0.j        ,  0.99999994+0.j        ]])


    '''
    pqr_ant_m = numpy.array(ant_pqr_m, dtype=numpy.float32)
    distances_pqr = near_field_distances(pqr_ant_m, source_pqr_m)
    ant_gains = numpy.array(
        numpy.exp(-2j*numpy.pi*freq_hz*distances_pqr/299792458.0),
        dtype=numpy.complex64)
    if not phase_only:
        ant_gains /= distances_pqr
    baseline_gains = ant_gains[:, numpy.newaxis, :]*numpy.conj(ant_gains[numpy.newaxis, :, :])
    num_ant = ant_pqr_m.shape[0]
    matrix_rows = []
    uvw_m = acm_as_vector(baseline_matrix_m(ant_pqr_m), include_diag=True)
    for row_id in range(num_ant):
        for col_id in range(row_id, num_ant):
            matrix_rows.append(baseline_gains[row_id, col_id, :])
    mat = numpy.array(matrix_rows, dtype=numpy.complex64)
    uv_lambda = numpy.sqrt(uvw_m[:,0]**2 + uvw_m[:,1]**2)*freq_hz/299792458.0
    bl_mask = numpy.ones(uvw_m.shape[0], dtype=numpy.int)
    if min_baseline_lambda is not None:
        bl_mask *= uv_lambda >= min_baseline_lambda
    if max_baseline_lambda is not None:
        bl_mask *= uv_lambda <= max_baseline_lambda

    return mat*bl_mask[:, numpy.newaxis]





class MatrixImager(object):
    def __init__(self, uvw_m, num_pixels, freq_hz,
                 min_baseline_lambda=None,
                 max_baseline_lambda=None):
        self.uvw_m = uvw_m.copy() 
        self.num_pixels = num_pixels
        self.freq_hz = freq_hz
        self.l_axis_rad = numpy.linspace(1.0, -1.0, num_pixels)
        self.m_axis_rad = numpy.linspace(-1.0, 1.0, num_pixels)
        self.imshow_extent = (1.0+1.0/(num_pixels-1), -1.0-1.0/(num_pixels-1),
                              -1.0-1.0/(num_pixels-1), 1.0+1.0/(num_pixels-1))
        #self.l_grid, self.m_grid = numpy.meshgrid(self.l_axis_rad, self.m_axis_rad)
        self.min_baseline_lambda = min_baseline_lambda
        self.max_baseline_lambda = max_baseline_lambda

        self.matrix, l_grid, m_grid = make_imaging_matrix(
            self.uvw_m, self.freq_hz,
            self.l_axis_rad, self.m_axis_rad,
            self.min_baseline_lambda,
            self.max_baseline_lambda)
        self.l_grid = l_grid.reshape((self.num_pixels, self.num_pixels))
        self.m_grid = m_grid.reshape((self.num_pixels, self.num_pixels))
        self.matrix /= (numpy.sum(numpy.abs(self.matrix))/self.num_pixels**2)       
        self.sky_mask = numpy.sqrt(self.l_grid**2+self.m_grid**2) > 1.0

        self.predict_matrix = None


    def dft_image(self, acm_vector, weights=None):
        n = self.num_pixels
        if weights is not None:
            matrix_weights = numpy.abs(self.matrix).mean(axis=0)
            w = weights*matrix_weights
            wsum = w.sum()
            wlen = len(w)
            mat = self.matrix*w[numpy.newaxis,:]*wlen/wsum
        else:
            mat = self.matrix
        return numpy.dot(mat, acm_vector).real.reshape((n, n))


    def compute_predict_matrix(self, source_lmn_rad, source_pqr_m=None, ant_pqr_m=None):
        r'''
        vis = matrix*source_fluxes
        '''
        predict_matrix =  make_predict_matrix(
            self.uvw_m, self.freq_hz, source_lmn_rad,
            self.min_baseline_lambda,
            self.max_baseline_lambda)
        if source_pqr_m is not None:
            nf_predict_matrix = predict_matrix_near_field(
                ant_pqr_m, self.freq_hz, source_pqr_m,
                min_baseline_lambda=self.min_baseline_lambda,
                max_baseline_lambda=self.max_baseline_lambda)
            predict_matrix = numpy.append(predict_matrix,
                                          nf_predict_matrix, axis=1)
        self.predict_matrix = predict_matrix
        return self.predict_matrix







class SkyModel(object):
    def __init__(self, pqr_to_itrs_matrix, observing_location_itrs,
                 sources=None):
        r'''
        sources: dict
            Name: ICRS SkyCoord.
        '''
        if sources is None:
            source_names = ['Cas A', 'Cyg A', 'Vir A', 'Her A', 'Tau A', 'Per A',
                            '3C 353', '3C 123', '3C 295', '3C 196', 'DR 4', 'DR 23', 'DR 21']
            source_icrs = [get_icrs_coordinates(source)
                           for source in source_names]
            self.sources = {name: icrs for name, icrs in zip(source_names, source_icrs)}
        else:
            self.sources = sources
        self.pqr_to_itrs_matrix = pqr_to_itrs_matrix
        xyz = observing_location_itrs
        self.obsgeoloc = acc.CartesianRepresentation(
            x=xyz[0], y=xyz[1], z=xyz[2], unit='m')



    def source_lmn_rad(self, obstime, include_sun=True):
        gcrs = acc.GCRS(obstime=obstime, obsgeoloc=self.obsgeoloc)
        gcrs_dict = {name: icrs.transform_to(gcrs)
                     for name, icrs in self.sources.items()}
        if include_sun:
            sun = acc.get_sun(obstime)
            gcrs_dict['Sun'] = sun.transform_to(gcrs)
        lmn_dict = {name: scc.pqr_from_icrs(
            numpy.array([gcrs.ra.rad, gcrs.dec.rad]),
            obstime=obstime,
            pqr_to_itrs_matrix=self.pqr_to_itrs_matrix)
                    for name, gcrs in gcrs_dict.items()}
        return lmn_dict


    def above_horizon(self, obstime, include_sun=True):
        lmn_dict = self.source_lmn_rad(obstime, include_sun)
        names = [name for name, lmn in lmn_dict.items() if lmn[2] >= 0.0]
        lmn = numpy.array([lmn_dict[name] for name in names])
        return names, lmn



def flux_density_jy(source, freq_hz, epoch=2017.5, scale='baars'):
    r'''
    scale = 'baars', 'perley', 'scaife-heald'
    '''
    if scale == 'baars':
        logfreq = numpy.log10(freq_hz/1e6)
        if source ==  'Cas A':
            sec_correction = (1-(0.97 -0.30*numpy.log10(freq_hz/1e9))/100)
            if freq_hz < 300e6:
                return (sec_correction**(epoch - 1980.0))*10**(5.745 + -0.770*logfreq)
            else:
                return (sec_correction**(epoch - 1965.0))*10**(5.625 -0.634*logfreq -0.023*logfreq**2)
        elif source == 'Cyg A':
            return 10**(4.695+0.085*logfreq -0.178*logfreq**2)
