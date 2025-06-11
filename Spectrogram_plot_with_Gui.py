
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for GUI compatibility
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                             AutoMinorLocator)
from matplotlib.colors import LogNorm
import pywt
import PySimpleGUI as sg
import os

def validate_numeric_input(values, key, min_val=None, max_val=None):
    try:
        value = float(values[key])
        if min_val is not None and value < min_val:
            return False, f"Value for {key} must be greater than {min_val}"
        if max_val is not None and value > max_val:
            return False, f"Value for {key} must be less than {max_val}"
        return True, value
    except ValueError:
        return False, f"Invalid numeric value for {key}"

def create_settings_window():
    # Available wavelet configurations
    wavelet_types = ['cmor1.0-1.0', 'cmor1.5-1.0', 'cmor2.0-1.0', 'cmor2.5-1.0']
    colormaps = ['jet', 'viridis', 'plasma', 'magma', 'inferno']
    layout = [
        [sg.Text('Select Input File:')],
        [sg.Input(key='-FILE-'), sg.FileBrowse(file_types=(('CSV Files', '*.csv'),))],
        [sg.Frame('Time Window Selection', [
            [sg.Text('Start Time (ms):'), sg.Input('0', key='-START_TIME-', size=(10, 1))],
            [sg.Text('End Time (ms):'), sg.Input('', key='-END_TIME-', size=(10, 1), tooltip='Leave empty to show full range')]
        ])],
        [sg.Frame('Wavelet Configuration', [
            [sg.Text('Wavelet Type:'), sg.Combo(wavelet_types, default_value='cmor2.0-1.0', key='-WAVELET-')],
            [sg.Text('Number of Scales:'), sg.Input('200', key='-SCALES-', size=(10, 1))],
            [sg.Text('Min Frequency (Hz):'), sg.Input('20000', key='-MIN_FREQ-', size=(10, 1))],
            [sg.Text('Max Frequency (Hz):'), sg.Input('500000', key='-MAX_FREQ-', size=(10, 1))],
            [sg.Text('Power Percentile:'), sg.Input('98', key='-POWER_PERC-', size=(10, 1))]
        ])],
        [sg.Frame('Visualization Settings', [
            [sg.Text('Colormap:'), sg.Combo(colormaps, default_value='jet', key='-COLORMAP-')],
            [sg.Text('Power Range:'), sg.Input('1000', key='-POWER_RANGE-', size=(10, 1))]
        ])],
        [sg.Button('Run Analysis'), sg.Button('Exit')]
    ]
    
    return sg.Window('Wavelet Analysis Settings', layout)

def run_analysis(file_path, wavelet_config, vis_config):
    try:
        plt.close('all')  # Close any existing plots
          # Show processing message
        processing_window = sg.Window('Processing', 
                                    [[sg.Text('Analyzing data... Please wait...')]])
        processing_window.Read(timeout=100)  # Show window briefly
        
        # Load and preprocess data
        df = pd.read_csv(file_path, usecols=[3, 4], names=["Time", "CH1"])
        header_info = pd.read_csv(file_path, nrows=2, header=None)
        
        # Extract sampling interval from header
        try:
            delta_time = float(header_info.iloc[1, 1])
            Fs = int(1/delta_time)
            if Fs <= 0:
                raise ValueError(f"Invalid sampling rate: {Fs} Hz")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error reading sampling rate from file: {str(e)}")

        # Process time series data
        time = df["Time"].values - df["Time"].values[0]
        data = (df["CH1"].values - np.mean(df["CH1"].values)) / np.std(df["CH1"].values)        # Set up frequency range
        nyquist_freq = Fs/2
        max_freq = min(nyquist_freq, wavelet_config['max_freq'])
        min_freq = max(wavelet_config['min_freq'], Fs/1000)

        # Calculate wavelet scales
        num_scales = wavelet_config['num_scales']
        central_freq = pywt.central_frequency(wavelet_config['wavelet'])
        # Adjust frequency range to account for wavelet central frequency
        adjusted_max_freq = max_freq / central_freq
        adjusted_min_freq = min_freq / central_freq
        frequencies = np.logspace(np.log10(adjusted_min_freq), np.log10(adjusted_max_freq), num_scales)
        scales = Fs / frequencies  # Remove central_freq multiplication since we adjusted frequencies

        # Filter scales
        valid_freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        scales = scales[valid_freq_mask]
        frequencies = frequencies[valid_freq_mask]

        # Perform wavelet transform
        coefficients, freqs = pywt.cwt(data, scales, wavelet_config['wavelet'], 
                                     sampling_period=delta_time)
        power = np.abs(coefficients) ** 2        # Handle time window selection
        start_time_ms = float(wavelet_config.get('start_time', 0))
        end_time_ms = time[-1]*1000 if wavelet_config.get('end_time') is None else float(wavelet_config['end_time'])
        start_idx = np.argmin(np.abs(time*1000 - start_time_ms))
        end_idx = np.argmin(np.abs(time*1000 - end_time_ms))
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8))
        
        # Time-domain plot
        ax1.set_title("Time-Domain Signal", fontsize=16)
        ax1.plot(time*1000, data, 'b-', linewidth=1)
        ax1.set_xlim(start_time_ms, end_time_ms)
        ax1.set_xlabel('Time (ms)', fontsize=14)
        ax1.set_ylabel('Amplitude (normalized)', fontsize=14)
        ax1.xaxis.set_tick_params(labelsize=12)
        ax1.yaxis.set_tick_params(labelsize=12)
        ax1.set_facecolor('whitesmoke')
        ax1.grid(True, alpha=0.5)
        
        # Wavelet spectrogram
        power_threshold = np.percentile(power, wavelet_config['power_percentile'])
        extent = [time[0]*1000, time[-1]*1000, frequencies.min(), frequencies.max()]
        im = ax2.imshow(power, aspect='auto', extent=extent, 
                       cmap=vis_config['colormap'], origin='lower',
                       norm=LogNorm(vmin=power_threshold/vis_config['power_range'], 
                                  vmax=power_threshold))
        
        ax2.set_title("Morlet Wavelet Spectrogram", fontsize=16)        
        ax2.set_xlabel('Time (ms)', fontsize=14)
        ax2.set_ylabel('Frequency (kHz)', fontsize=14)
        ax2.set_yscale('log')
        ax2.set_ylim(min_freq, max_freq)
        ax2.set_xlim(start_time_ms, end_time_ms)
        ax2.yaxis.set_major_formatter(lambda x, pos: f'{x/1000:.0f}')        # Create colorbar below the spectrogram
        ax2.xaxis.set_tick_params(labelsize=12)
        ax2.yaxis.set_tick_params(labelsize=12)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("bottom", size="5%", pad=0.6)  # Increased pad to move colorbar down
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label('Power (normalized)', labelpad=0.1)  # Added labelpad to move label down
        

        # Add configuration information
        config_text = (
            f"Wavelet: {wavelet_config['wavelet']}\n"
            f"Scales: {wavelet_config['num_scales']}\n"
            f"Freq Range: {wavelet_config['min_freq']/1000:.1f}-{wavelet_config['max_freq']/1000:.1f} kHz\n"
            f"Power Percentile: {wavelet_config['power_percentile']}%\n"
            f"Power Range: {vis_config['power_range']}\n"
            f"Colormap: {vis_config['colormap']}"
        )
        # Add text box with configuration
        plt.figtext(0.02, 0.02, config_text, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        plt.tight_layout()
        # Adjust spacing to align x-axes and add right padding
        plt.subplots_adjust(hspace=0.3)  # Added right padding
        processing_window.close()
        plt.show(block=False)  # Non-blocking plot display
        return True

    except Exception as e:
        if 'processing_window' in locals():
            processing_window.close()
        sg.popup_error(f'Error during analysis: {str(e)}')
        return False

def main():
    # Set up theme and options
    sg.change_look_and_feel('DefaultNoMoreNagging')
    sg.SetOptions(element_padding=(0, 0))
    sg.set_options(dpi_awareness=True)
    
    window = create_settings_window()
    try:
        while True:
            event, values = window.Read()  # Capital R in older versions
            
            if event is None or event == 'Exit':  # Different event checking in older version
                break
                
            if event == 'Run Analysis':
                # Validate file selection
                if not values['-FILE-'] or not os.path.exists(values['-FILE-']):
                    sg.popup_error('Please select a valid input file!')
                    continue                # Validate numeric inputs
                validation_checks = [
                    ('-SCALES-', 10, 1000),
                    ('-MIN_FREQ-', 1, None),
                    ('-MAX_FREQ-', None, None),
                    ('-POWER_PERC-', 0, 100),
                    ('-POWER_RANGE-', 1, None),
                    ('-START_TIME-', 0, None)
                ]
                
                # Validate end time if provided
                if values['-END_TIME-']:
                    validation_checks.append(('-END_TIME-', float(values['-START_TIME-']), None))

                valid = True                
                for key, min_val, max_val in validation_checks:
                    is_valid, result = validate_numeric_input(values, key, min_val, max_val)
                    if not is_valid:
                        sg.popup_error(result)
                        valid = False
                        break

                if not valid:
                    continue
                
                # Configure analysis parameters
                wavelet_config = {
                    'wavelet': values['-WAVELET-'],
                    'num_scales': int(float(values['-SCALES-'])),
                    'min_freq': float(values['-MIN_FREQ-']),
                    'max_freq': float(values['-MAX_FREQ-']),
                    'power_percentile': float(values['-POWER_PERC-']),
                    'start_time': float(values['-START_TIME-']),
                    'end_time': float(values['-END_TIME-']) if values['-END_TIME-'].strip() else None
                }
                vis_config = {
                    'colormap': values['-COLORMAP-'],
                    'power_range': float(values['-POWER_RANGE-'])
                }
                window.Hide()  # Capital H in older version
                success = run_analysis(values['-FILE-'], wavelet_config, vis_config)
                if not success:
                    window.UnHide()  # Capital U and H in older version
                else:                    # Create a control window for the plot
                    control_layout = [
                        [sg.Text('What would you like to do?', size=(30, 1), justification='center')],
                        [sg.Button('New Analysis', size=(15, 1)), sg.Button('Exit', size=(15, 1))]
                    ]
                    control_window = sg.Window('Analysis Control', 
                                            control_layout,
                                            size=(500, 100),
                                            element_justification='center',
                                            finalize=True,
                                            keep_on_top=True,
                                            grab_anywhere=True)  # Allows window to be moved
                    
                    control_event, _ = control_window.Read()  # Capital R in older version
                    plt.close('all')
                    control_window.Close()  # Capital C in older version
                    
                    if control_event == 'New Analysis':
                        window.UnHide()
                    else:
                        break

    finally:
        if window is not None:
            window.close()

if __name__ == '__main__':
    main()