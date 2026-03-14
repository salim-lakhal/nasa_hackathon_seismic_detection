"""
Spectrogram generation utilities for seismic data.
Extracted and cleaned from notebook code with adaptive FFT-based filtering.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
from matplotlib import cm
from obspy import read
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpectrogramGenerator:
    """Generate spectrograms from seismic miniseed files with adaptive filtering."""

    def __init__(self, output_dir=None, img_size=(224, 224), dpi=300):
        """
        Initialize spectrogram generator.

        Args:
            output_dir: Directory to save spectrograms
            img_size: Output image size (width, height)
            dpi: Resolution for saved images
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.img_size = img_size
        self.dpi = dpi

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_adaptive_filter_params(self, trace_data, sampling_rate):
        """
        Compute adaptive bandpass filter parameters based on FFT analysis.

        Args:
            trace_data: Seismic trace data array
            sampling_rate: Sampling rate in Hz

        Returns:
            dict: Filter parameters including minfreq, maxfreq, dominant frequency
        """
        n = len(trace_data)
        T = 1.0 / sampling_rate

        # Compute FFT
        yf = fft(trace_data)
        xf = np.fft.fftfreq(n, T)[:n // 2]
        spectrum = 2.0 / n * np.abs(yf[0:n // 2])

        # Find dominant frequency
        max_index = np.argmax(spectrum)
        max_frequency = xf[max_index]
        max_amplitude = spectrum[max_index]

        # Compute Full Width at Half Maximum (FWHM)
        half_max = max_amplitude / 2
        indices_above_half_max = np.where(spectrum >= half_max)[0]

        if len(indices_above_half_max) > 0:
            fwhm = xf[indices_above_half_max[-1]] - xf[indices_above_half_max[0]]
        else:
            fwhm = 0

        # Calculate percentage for filter span
        percentage = (fwhm / max_frequency) * 100 if max_frequency != 0 else 5
        percentage = max(percentage, 5)  # Minimum 5% span

        # Define bandpass filter range
        span = (percentage / 100) * max_frequency
        minfreq = max(max_frequency - span, 0.0001)  # Avoid zero or negative
        maxfreq = max_frequency + span

        return {
            'minfreq': minfreq,
            'maxfreq': maxfreq,
            'dominant_freq': max_frequency,
            'fwhm': fwhm,
            'percentage': percentage
        }

    def generate_spectrogram(self, mseed_file, output_path=None, show_plot=False):
        """
        Generate spectrogram from miniseed file with adaptive filtering.

        Args:
            mseed_file: Path to miniseed file
            output_path: Optional custom output path
            show_plot: Whether to display the plot

        Returns:
            dict: Spectrogram metadata and path
        """
        try:
            # Read seismic data
            st = read(str(mseed_file))
            tr = st.traces[0].copy()
            tr_data = tr.data
            sampling_rate = tr.stats.sampling_rate

            # Compute adaptive filter parameters
            filter_params = self.compute_adaptive_filter_params(tr_data, sampling_rate)

            logger.info(f"Processing {Path(mseed_file).name}")
            logger.info(f"  Dominant frequency: {filter_params['dominant_freq']:.4f} Hz")
            logger.info(f"  Filter range: {filter_params['minfreq']:.4f} - {filter_params['maxfreq']:.4f} Hz")

            # Apply bandpass filter
            st_filt = st.copy()
            st_filt.filter('bandpass',
                          freqmin=filter_params['minfreq'],
                          freqmax=filter_params['maxfreq'])
            tr_filt = st_filt.traces[0].copy()
            tr_data_filt = tr_filt.data

            # Generate spectrogram
            f, t, sxx = signal.spectrogram(tr_data_filt, sampling_rate)

            # Create figure without axes for clean ML input
            fig = plt.figure(figsize=(10, 6))
            ax = plt.subplot(1, 1, 1)
            vals = ax.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
            ax.set_xlim([t.min(), t.max()])
            ax.set_axis_off()

            # Determine output path
            if output_path is None and self.output_dir:
                filename = Path(mseed_file).stem
                output_path = self.output_dir / f"{filename}.png"

            # Save spectrogram
            if output_path:
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0)
                logger.info(f"  Saved to: {output_path}")

            if show_plot:
                plt.show()
            else:
                plt.close()

            return {
                'filename': str(mseed_file),
                'output_path': str(output_path) if output_path else None,
                'sampling_rate': sampling_rate,
                'filter_params': filter_params,
                'spectrogram_shape': sxx.shape
            }

        except Exception as e:
            logger.error(f"Error processing {mseed_file}: {e}")
            raise

    def batch_generate(self, mseed_files, output_dir=None):
        """
        Generate spectrograms for multiple miniseed files.

        Args:
            mseed_files: List of miniseed file paths
            output_dir: Directory to save spectrograms

        Returns:
            list: List of metadata dicts for each generated spectrogram
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for mseed_file in mseed_files:
            try:
                result = self.generate_spectrogram(mseed_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {mseed_file}: {e}")
                continue

        logger.info(f"Successfully generated {len(results)}/{len(mseed_files)} spectrograms")
        return results
