from numpy import cos, sin, arcsin, arctan2, sqrt, array, dot, float64
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
import astropy.units as u

# Always try to download the most recent IERS tables.
from astropy.utils.data import download_file
from astropy.utils import iers
iers.IERS.iers_table = iers.IERS_A.open(download_file(iers.IERS_A_URL, cache=True))



def lm_from_radec(ra, dec, ra0, dec0):
    r'''
    Calculate direction cosines given a right ascension and
    declination and the right ascension and declination of the
    projection centre.

    **Parameters**

    ra : astropy.coordinates.Angle
        Right ascension to convert.

    dec : astropy.coordinates.Angle
        Declination to convert.

    ra0 : astropy.coordinates.Angle
        Right ascension of the projection centre.

    dec0 : astropy.coordinates.Angle
        Declination of the projection centre.


    **Returns**

    A tuple of floats (l_rad, m_rad).


    **Examples**

    >>> import astropy.units as u
    >>> lm_from_radec(ra=Angle(2.0, u.rad), dec=Angle(1.1, u.rad),
    ...               ra0=Angle(2.0, u.rad), dec0=Angle(1.0, u.rad))
    (0.0, 0.099833416646828321)

    >>> lm_from_radec(ra=Angle(2.2, u.rad), dec=Angle(0.0, u.rad),
    ...               ra0=Angle(2.0, u.rad), dec0=Angle(0.0, u.rad))
    (0.19866933079506138, 0.0)

    >>> lm_from_radec(ra=Angle(2.0, u.rad), dec=Angle(1.0, u.rad),
    ...               ra0=Angle(2.0, u.rad), dec0=Angle(1.0, u.rad))
    (0.0, 0.0)

    >>> lm_from_radec(ra=Angle(1.8, u.rad), dec=Angle(0.9, u.rad),
    ...               ra0=Angle(2.0, u.rad), dec0=Angle(1.0, u.rad))
    (-0.1234948364118721, -0.089406906258670149)
    '''
    cos_dec  = cos(dec.rad)
    sin_dec  = sin(dec.rad)
    cos_dec0 = cos(dec0.rad)
    sin_dec0 = sin(dec0.rad)
    sin_dra  = sin(float(ra.rad - ra0.rad))
    cos_dra  = cos(float(ra.rad - ra0.rad))

    l_rad = cos_dec*sin_dra
    m_rad = sin_dec*cos_dec0 - cos_dec*sin_dec0*cos_dra
    return (l_rad, m_rad)





def radec_from_lm(l_rad, m_rad, ra0, dec0):
    r'''
    Calculate right ascension and declination given direction cosines
    l and m and the RA and Dec of the projection centres.

    **Parameters**

    l_rad : float
        Direction cosine parallel to right ascension at the projection
        centre (in rad).

    m_rad : float
        Direction cosine parallel to declination at the projection
        centre (in rad).

    ra0 : astropy.coordinates.Angle
        Right ascension of the projection centre.

    dec0 : astropy.coordinates.Angle
        Declination of the projection centre.


    **Returns**

    A tuple of astropy.coordinates.Angles (ra, dec)


    **Examples**

    >>> import astropy.units as u
    >>> radec_from_lm(l_rad=0.0, m_rad=0.099833416646828321,
    ...               ra0=Angle(2.0, u.rad), dec0=Angle(1.0, u.rad))
    (<Angle 2.0 rad>, <Angle 1.1 rad>)

    >>> radec_from_lm(l_rad=0.19866933079506138, m_rad=0.0,
    ...               ra0=Angle(2.0, u.rad), dec0=Angle(0.0, u.rad))
    (<Angle 2.2 rad>, <Angle 0.0 rad>)
    >>> radec_from_lm(l_rad=-0.1234948364118721, m_rad=-0.089406906258670149,
    ...               ra0=Angle(2.0, u.rad), dec0=Angle(1.0, u.rad))
    (<Angle 1.8 rad>, <Angle 0.9 rad>)

    '''
    n_rad  = sqrt(1.0 - l_rad*l_rad - m_rad*m_rad)
    cos_dec0 = cos(dec0.rad)
    sin_dec0 = sin(dec0.rad)
    ra_rad = ra0.rad + arctan2(l_rad, cos_dec0*n_rad - m_rad*sin_dec0)
    dec_rad = arcsin(m_rad*cos_dec0 + sin_dec0*n_rad)
    return (Angle(ra_rad, u.rad), Angle(dec_rad, u.rad))





def icrs_from_itrs(unit_vector_itrs, obstime):
    r'''
    Convert a geocentric cartesian unit vector in the ITRS system into
    an astropy.coordinates.SkyCoord in the ICRS system (equinox
    J2000), given an observing date/time. This routine is used to find
    out the RA and DEC of the direction to which a station's normal
    vector points.

    **Parameters**
    
    unit_vector_itrs : numpy array of 3 floats
        The direction to convert it is assumed to be in the ITRS
        system.

    obstime : astropy.time.Time or string
        When a string is provided, it is assumed to be readable by an
        astropy.time.Time instance.


    **Returns**

    A SkyCoord containing the ICRS position.


    **Examples**

    >>> icrs_from_itrs([0.0, 0.0, 1.0], '2000-01-01 00:00:00')
    <SkyCoord (ICRS): (ra, dec) in deg
        ( 349.95493176,  89.99588845)>

    >>> icrs_from_itrs([0.0, 1.0, 0.0], '2000-03-21 12:00:00')
    <SkyCoord (ICRS): (ra, dec) in deg
        ( 89.31087327,  0.00329332)>

    >>> icrs_from_itrs([0.0, 0.0, 1.0], '2015-01-01 00:00:00')
    <SkyCoord (ICRS): (ra, dec) in deg
        ( 358.79289756,  89.91033405)>
    '''
    x, y, z = array(unit_vector_itrs).T
    c_itrs = SkyCoord(x, y, z, representation='cartesian',frame='itrs',
                      obstime=obstime, equinox='J2000')
    return c_itrs.icrs





def itrs_from_icrs(icrs_position_rad, obstime):
    r'''
    Takes an array RA/Dec ICRS positions in radians and converts those
    to geocentric ITRF unit vectors.
    
    **Parameters**

    icrs_position_rad : numpy array of floats (ra_rad, dec_rad)
        The ICRS position to convert. May also be an array of RA/DEC
        pairs.

    obstime : astropy.time.Time or string
        When a string is provided, it is assumed to be readable by an
        astropy.time.Time instance.

    **Returns**

    An array containing the geocentric cartesian ITRS unit vectors
    corresponding to the icrs_position_rad at obstime.

    **Examples**

    >>> itrs_from_icrs((array([358.7928571, 89.91033405])*u.deg).to(u.rad),
    ...                obstime='2015-01-01 00:00:00')
    array([ -1.06671307e-09,   3.01033878e-10,   1.00000000e+00])

    >>> itrs_from_icrs((array([[358.7928571, 89.91033405],
    ...                        [90,-20],
    ...                        [30, 60]])*u.deg).to(u.rad),
    ...                        obstime='2015-01-01 00:00:00')
    array([[ -1.06671307e-09,   3.01033878e-10,   1.00000000e+00],
           [  9.24936412e-01,  -1.65757654e-01,  -3.42077527e-01],
           [  1.70173410e-01,  -4.68931931e-01,   8.66685557e-01]])

    '''
    ra, dec = array(icrs_position_rad, dtype='float64').T*u.rad
    icrs = SkyCoord(ra, dec,frame='icrs',
                    obstime=obstime, equinox='J2000')
    itrs = icrs.itrs
    return array([itrs.x, itrs.y, itrs.z], dtype=float64).T




def pqr_from_icrs(icrs_rad, obstime, pqr_to_itrs_matrix):
    r'''
    Compute geocentric station-local PQR coordinates of a certain ICRS
    direction. Geocentric here means that parallax between the centre
    of the Earth and the station's phase centre is not taken into
    account.

    **Parameters**

    icrs_rad : numpy.array
        An ICRS position in radians. Either one vector of length 2, or
        an array of shape (N, 2) containing N ICRS [RA, Dec] pairs.

    obstime : string or astropy.time.Time
        The date/time of observation.

    pqr_to_itrs_matrix : 3x3 numpy.array
        The rotation matrix that is used to convert a direction in the
        station's PQR system into an ITRS direction. This matrix is
        found in the /opt/lofar/etc/AntennaField.conf files at the
        stations. These files are also found in the antenna-fields/
        directory of this project.

    **Returns**
    
    A numpy.array instance with shape (3,) or (N, 3) containing the
    ITRS directions.

    **Example**
    
    >>> import astropy
    >>> core_pqr_itrs_mat = array([[ -1.19595000e-01,  -7.91954000e-01,   5.98753000e-01],
    ...                            [  9.92823000e-01,  -9.54190000e-02,   7.20990000e-02],
    ...                            [  3.30000000e-05,   6.03078000e-01,   7.97682000e-01]],
    ...                           dtype=float64)
    >>> obstime='2015-06-19 13:50:00'
    >>> target_3c196 = SkyCoord('08h13m36.0561s', '+48d13m02.636s', frame='icrs')
    >>> target_icrs_rad = array([target_3c196.icrs.ra.rad, target_3c196.icrs.dec.rad])
    >>> pqr_from_icrs(target_icrs_rad, obstime, core_pqr_itrs_mat)
    array([ 0.0213451 , -0.0823542 ,  0.99637451])
    >>> pqr_from_icrs(array([target_icrs_rad, target_icrs_rad]),
    ...                      obstime, core_pqr_itrs_mat)
    array([[ 0.0213451 , -0.0823542 ,  0.99637451],
           [ 0.0213451 , -0.0823542 ,  0.99637451]])
    >>> pqr_from_icrs(target_icrs_rad, astropy.time.Time(obstime)+7*u.minute +18*u.second,
    ...               core_pqr_itrs_mat)
    array([  4.74959541e-05,  -8.26257556e-02,   9.96580637e-01])

    '''
    return dot(pqr_to_itrs_matrix.T, itrs_from_icrs(icrs_rad, obstime).T).T.squeeze()




def icrs_from_pqr(pqr, obstime, pqr_to_itrs_matrix):
    r'''
    Convert directions from the station-local PQR system into
    geocentric ICRS position.
    
    **Parameters**
    
    pqr : numpy array
        The direction to convert it is assumed to be in the ITRS
        system. Either a numpy.array of 3 floats, or an (N,3)
        numpy.array contining a collection of PQR positions to
        convert.

    obstime : astropy.time.Time or string
        When a string is provided, it is assumed to be readable by an
        astropy.time.Time instance.
    
    pqr_to_itrs_matrix : 3x3 numpy.array
        The rotation matrix that is used to convert a direction in the
        station's PQR system into an ITRS direction. This matrix is
        found in the /opt/lofar/etc/AntennaField.conf files at the
        stations. These files are also found in the antenna-fields/
        directory of this project.

    **Returns**

    A SkyCoord containing the corresponding ICRS position(s).

    **Examples**
    
    >>> import astropy
    >>> core_pqr_itrs_mat = array([[ -1.19595000e-01,  -7.91954000e-01,   5.98753000e-01],
    ...                            [  9.92823000e-01,  -9.54190000e-02,   7.20990000e-02],
    ...                            [  3.30000000e-05,   6.03078000e-01,   7.97682000e-01]],
    ...                           dtype=float64)
    >>> obstime='2015-06-19 13:50:00'
    >>> icrs_from_pqr(array([[0, -1, 0], [0, 0, 1], [0, 1, 0]]),
    ...               obstime, core_pqr_itrs_mat)
    <SkyCoord (ICRS): (ra, dec) in deg
        [( 121.70156757, -37.04094451), ( 121.55013425,  52.95456305),
         ( 301.68992195,  37.04486895)]>

    >>> icrs_from_pqr(array([[0, -1, 0], [0, 0, 1], [0, 1, 0]]),
    ...               astropy.time.Time(obstime)+4*u.minute, core_pqr_itrs_mat)
    <SkyCoord (ICRS): (ra, dec) in deg
        [( 122.70359536, -37.03964509), ( 122.55389784,  52.95575172),
         ( 302.69208851,  37.04366637)]>

    >>> source_3c196 = SkyCoord('08h13m36.0561s', '+48d13m02.636s', frame='icrs')
    >>> pqr = pqr_from_icrs([source_3c196.icrs.ra.rad, source_3c196.icrs.dec.rad],
    ...                     obstime, core_pqr_itrs_mat)
    >>> pqr
    array([ 0.0213451 , -0.0823542 ,  0.99637451])
    >>> icrs_from_pqr(pqr, obstime, core_pqr_itrs_mat)
    <SkyCoord (ICRS): (ra, dec) in deg
        ( 123.40023896,  48.21740382)>
    
    Let's check if that is indeed 3C 196:
    >>> source_3c196
    <SkyCoord (ICRS): (ra, dec) in deg
        ( 123.40023375,  48.21739889)>
    
    And it is not. The reason for this is the limited precision with
    which the rotation matrices are listed in the AntennaField.conf
    files. Sure enough:

    >>> dot(core_pqr_itrs_mat, core_pqr_itrs_mat.T)
    array([[  9.99999257e-01,   2.84588000e-07,   5.09499000e-07],
           [  2.84588000e-07,   1.00000056e+00,  -2.62005000e-07],
           [  5.09499000e-07,  -2.62005000e-07,   9.99999648e-01]])
    
    A 2e-7 differences correspond to about 1e-5 degrees, which isthe
    level at which we see the difference.
    '''
    return icrs_from_itrs(dot(pqr_to_itrs_matrix, pqr.T).T, obstime=obstime)
    
