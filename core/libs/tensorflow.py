
import tensorflow
from tensorflow.image import ResizeMethod
from tensorflow import dtypes
from tensorflow.python.ops import math_ops, array_ops, gen_image_ops
from tensorflow.python.ops.image_ops_impl import _resize_images_common
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export('image.resize', v1=[])
@dispatch.add_dispatch_support
def resize(images,
           size,
           method=ResizeMethod.BILINEAR,
           preserve_aspect_ratio=False,
           antialias=False,
           name='resize2',
           half_pixel_centers=False):

    def resize_fn(images_t, new_size):
        """Resize core function, passed to _resize_images_common."""
        scale_and_translate_methods = [
            ResizeMethod.LANCZOS3, ResizeMethod.LANCZOS5, ResizeMethod.GAUSSIAN,
            ResizeMethod.MITCHELLCUBIC
        ]

        def resize_with_scale_and_translate(method):
            scale = (
                math_ops.cast(new_size, dtype=dtypes.float32) /
                math_ops.cast(array_ops.shape(images_t)[1:3], dtype=dtypes.float32))
            return gen_image_ops.scale_and_translate(
                images_t,
                new_size,
                scale,
                array_ops.zeros([2]),
                kernel_type=method,
                antialias=antialias)

        if method == ResizeMethod.BILINEAR:
            if antialias:
                return resize_with_scale_and_translate('triangle')
            else:
                return gen_image_ops.resize_bilinear(
                    images_t, new_size, half_pixel_centers=half_pixel_centers)
        elif method == ResizeMethod.NEAREST_NEIGHBOR:
            return gen_image_ops.resize_nearest_neighbor(
                images_t, new_size, half_pixel_centers=half_pixel_centers)
        elif method == ResizeMethod.BICUBIC:
            if antialias:
                return resize_with_scale_and_translate('keyscubic')
            else:
                return gen_image_ops.resize_bicubic(
                    images_t, new_size, half_pixel_centers=True)
        elif method == ResizeMethod.AREA:
            return gen_image_ops.resize_area(images_t, new_size)
        elif method in scale_and_translate_methods:
            return resize_with_scale_and_translate(method)
        else:
            raise ValueError(
                'Resize method is not implemented: {}'.format(method))

    return _resize_images_common(
        images,
        resize_fn,
        size,
        preserve_aspect_ratio=preserve_aspect_ratio,
        name=name,
        skip_resize_if_same=False)
