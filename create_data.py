import LION.CTtools.ct_utils as ct
from ts_algorithms import fbp, tv_min2d
import LION.CTtools.ct_geometry as ctgeo



def create_sinograms(images, angles_full):
    # 0.1: Make geometry:
    geo = ctgeo.Geometry.parallel_default_parameters(
        image_shape=images.shape, number_of_angles=angles_full
    )  # parallel beam standard CT
    # 0.2: create operator:
    op = ct.make_operator(geo)
    # 0.3: forward project:
    sino = op(torch.from_numpy(images))
    sinogram_full = torch.moveaxis(sinogram_full, -1, -2)
    return np.asarray(sinogram_full.unsqueeze(1))

