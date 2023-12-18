

from .devito import SpongeBoundary1, SpongeBoundary2, ComplexFrequencyShiftPML2

boundaries_registry = {
    'sponge_boundary_1': SpongeBoundary1,
    'sponge_boundary_2': SpongeBoundary2,
    'complex_frequency_shift_PML_2': ComplexFrequencyShiftPML2,
}
