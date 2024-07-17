from collections import namedtuple

WrapNBRKResult = namedtuple(
     'WrapNBRKResult',
        (
            'success',
            'message',
            'size',
            'num_dy',
            'num_y',
            'error_code',
            'y',
            't'
        )
    )
