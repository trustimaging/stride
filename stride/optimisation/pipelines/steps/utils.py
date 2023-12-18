
from mosaic.utils import snake_case


def name_from_op_name(op, obj):
    return '%s_%s' % (snake_case(op.__class__.__name__), obj.name)
