import numpy as np

measurement_defaults = {'linestyle': 'none',
                        'marker': '.',
                        'markersize': 2,
                        'color': 'gray',
                        'alpha': 1}

model_defaults = {'linestyle': '-',
                  'linewidth': 0.3,
                  'color': 'C0',
                  'alpha': 1}

resonance_defaults = {'linestyle': 'none',
                      'marker': '.',
                      'markersize': 3,
                      'color': 'C1',
                      'alpha': 1}


def resonator_amplitude(resonator, axis, normalize=False, num_model_points=10000, frequency_scale=1e-6, three_ticks=True,
                        decibels=True, measurement_settings=None, model_settings=None, resonance_settings=None):
    measurement_kwds = measurement_defaults.copy()
    if measurement_settings is not None:
        measurement_kwds.update(measurement_settings)
    model_kwds = model_defaults.copy()
    if model_settings is not None:
        model_kwds.update(model_settings)
    resonance_kwds = resonance_defaults.copy()
    if resonance_settings is not None:
        resonance_kwds.update(resonance_settings)
    rd = resonator.measurement_model_resonance(normalize=normalize, num_model_points=num_model_points)
    if decibels:
        data_transform = lambda data: 20 * np.log10(np.abs(data))
    else:
        data_transform = lambda data: np.abs(data)
    axis.plot(frequency_scale * rd.measurement_frequency, data_transform(rd.measurement_data), **measurement_kwds)
    axis.plot(frequency_scale * rd.model_frequency, data_transform(rd.model_data), **model_kwds)
    axis.plot(frequency_scale * rd.resonance_frequency, data_transform(rd.resonance_data), **resonance_kwds)
    if three_ticks:
        axis.set_xticks(frequency_scale * np.array([rd.measurement_frequency.min(), rd.resonance_frequency,
                                                    rd.measurement_frequency.max()]))
    return rd


def resonator_phase(resonator, axis, normalize=False, num_model_points=10000, frequency_scale=1e-6, three_ticks=True,
                    degrees=True, measurement_settings=None, model_settings=None, resonance_settings=None):
    measurement_kwds = measurement_defaults.copy()
    if measurement_settings is not None:
        measurement_kwds.update(measurement_settings)
    model_kwds = model_defaults.copy()
    if model_settings is not None:
        model_kwds.update(model_settings)
    resonance_kwds = resonance_defaults.copy()
    if resonance_settings is not None:
        resonance_kwds.update(resonance_settings)
    rd = resonator.measurement_model_resonance(normalize=normalize, num_model_points=num_model_points)
    if degrees:
        data_transform = lambda data: np.degrees(np.angle(data))
    else:
        data_transform = lambda data: np.angle(data)
    axis.plot(frequency_scale * rd.measurement_frequency, data_transform(rd.measurement_data), **measurement_kwds)
    axis.plot(frequency_scale * rd.model_frequency, data_transform(rd.model_data), **model_kwds)
    axis.plot(frequency_scale * rd.resonance_frequency, data_transform(rd.resonance_data), **resonance_kwds)
    if three_ticks:
        axis.set_xticks(frequency_scale * np.array([rd.measurement_frequency.min(), rd.resonance_frequency,
                                                    rd.measurement_frequency.max()]))
    return rd


def resonator_complex_plane(resonator, axis, normalize=False, num_model_points=10000,
                            measurement_settings=None, model_settings=None, resonance_settings=None):
    measurement_kwds = measurement_defaults.copy()
    if measurement_settings is not None:
        measurement_kwds.update(measurement_settings)
    model_kwds = model_defaults.copy()
    if model_settings is not None:
        model_kwds.update(model_settings)
    resonance_kwds = resonance_defaults.copy()
    if resonance_settings is not None:
        resonance_kwds.update(resonance_settings)
    rd = resonator.measurement_model_resonance(normalize=normalize, num_model_points=num_model_points)
    axis.plot(rd.measurement_data.real, rd.measurement_data.imag, **measurement_kwds)
    axis.plot(rd.model_data.real, rd.model_data.imag, **model_kwds)
    axis.plot(rd.resonance_data.real, rd.resonance_data.imag, **resonance_kwds)
    return rd
