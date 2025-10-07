import PySimpleGUI as sg
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import re
import subprocess
import ffmpeg

def create_test_window():
    layout = [
        [sg.Text('What do you want to do?')],
        [sg.Button('Update'), sg.Button('Export'), sg.Button('Import'), sg.Button('Exit')]
    ]
    window = sg.Window('Test Window', layout)
    return window

def Update_window():
    # Load Excel sheet
    excel_file = 'Dataset_Options.xlsx'
    df = pd.read_excel(excel_file)

    # Extract unique values from each column for drop-down options
    track_options = df['Tracks'].dropna().unique().tolist()
    condition_options = df['Condition'].dropna().unique().tolist()
    psi_options = df['PSI'].dropna().unique().tolist()
    # Years (from 2025 to 2050)
    years = [str(y) for y in range(2025, 2051)]
    # Months (1-12)
    months = [str(m).zfill(2) for m in range(1, 13)]
    # Days (1-31)
    days = [str(d).zfill(2) for d in range(1, 32)]

    layout = [
        [sg.Text('Update Window')],
        [sg.Text("Pick a Track:")],
        [sg.Combo(track_options, default_value=track_options[0] if track_options else '', key='track', readonly=True)],
        [sg.Text("Pick a Condition:")],
        [sg.Combo(condition_options, default_value=condition_options[0] if condition_options else '', key='condition', readonly=True)],
        [sg.Text("Pick PSI:")],
        [sg.Combo(psi_options, default_value=psi_options[1] if psi_options else '', key='psi', readonly=True)],
        [sg.Text("Choose an MP4 video:"), sg.Input(), sg.FileBrowse(key='Vid_In', file_types=(("MP4 Videos", "*.mp4"),))],
        [sg.Text("What is the start time (in seconds)?"), sg.Input(key='start_time', size=(10,1))],
        [sg.Text("What is the end time (in seconds)?"), sg.Input(key='end_time', size=(10,1))],
        [sg.Text("What is the Ground Moisture Content(%)?"), sg.Combo(['6%', '7%', '8%', '9%', '10%', '11%', '12%', '13%', '14%', '15%','16%','17%', '18%', '19%'], default_value='6%', key='GMC', readonly=True)],
        [sg.Text("Each percent includes up to the next 0.9% (e.g., 6% includes 6.0% to 6.9%)")],
        [sg.Text("When was the data collected? (MM/DD/YYYY)")],
        [sg.Combo(months, default_value=months[0], key='month', readonly=True),
         sg.Combo(days, default_value=days[0], key='day', readonly=True),
         sg.Combo(years, default_value=years[0], key='year', readonly=True)],
        [sg.Button('OK'), sg.Button('Cancel')]
    ]

    window = sg.Window('Update', layout)
    return window

def Import_window():
    layout = [
        [sg.Text('Import Window')],
        [sg.Text('This is where import functionality would go.')],
        [sg.Button('Back')]
    ]
    window = sg.Window('Import', layout)
    return window

def Export_window():
    layout = [
        [sg.Text('Export Window')],
        [sg.Text('This is where export functionality would go.')],
        [sg.Button('Back')]
    ]
    window = sg.Window('Export', layout)
    return window

def cut_video(input_path, output_path, start_time, end_time):
    (
        ffmpeg
        .input(input_path, ss=start_time, to=end_time, r=2)  # Set frame rate to 2 fps
        .output(f'output_path')  
        .run(overwrite_output=True)
    )



def main():
    window = create_test_window()
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        elif event == 'Update':
            window.close()
            window = Update_window()
            event, values = window.read()
            if event == 'OK':
                window.close()
                PNG_Ref_excel = 'PNG_Ref.xlsx'
                png_ref = pd.read_excel(PNG_Ref_excel) # Load the PNG reference Excel file
                marker = 1 #To handle repeats of the same track on the same day
                names = png_ref['Track_S'].unique().tolist()
                filename = f"{values['month']}{values['day']}{values['year']}_{values['track']}_{marker}_"
                while True:
                    if filename in names:
                        marker += 1
                        filename = f"{values['month']}{values['day']}{values['year']}_{values['track']}_{marker}_"
                    else:
                        break
                filename = f"{values['month']}{values['day']}{values['year']}_{values['track']}_{marker}_"
                print(f"Generated filename: {filename}")
                # Create hash tables for each string column
                condition_map = {row['Condition_S']: row['Condition'] for _, row in png_ref.iterrows()}
                psi_map = {row['PSI_S']: row['PSI'] for _, row in png_ref.iterrows()}
                track_map = {row['Track_S']: row['Track'] for _, row in png_ref.iterrows()}
                new_row = [filename, condition_map.get(values['condition']), psi_map.get(values['psi']), track_map.get(values['track']), values['condition'], values['psi'], values['track']]
                new_row_png = pd.DataFrame([new_row], columns=png_ref.columns)
                png_ref = pd.concat([png_ref, new_row_png], ignore_index=True)
                png_ref.to_excel(PNG_Ref_excel, index=False)
                print(f"Updated {PNG_Ref_excel} with new entry.")
                # Cut the video using ffmpeg    
                os.makedirs('Temp_frames', exist_ok=True)
                input_video_path = values['Vid_In']
                output_video_path = os.path.join('Temp_frames', f"{filename}_frame%04d.png")
                cut_video(input_video_path, output_video_path, values['start_time'], values['end_time'])
                frame_files = sorted(os.listdir('Temp_frames'))
                fr = pd.read_excel('Folder_ref.xlsx')
                # Create dictionaries mapping Input to Train_fold and Test_fold
                train_folder_map = {row['Input']: row['Train_fold'] for _, row in fr.iterrows()}
                test_folder_map = {row['Input']: row['Test_fold'] for _, row in fr.iterrows()}
                output_folder_Train = train_folder_map.get(values['GMC'])
                output_folder_Test = test_folder_map.get(values['GMC'])
                for idx, filename in enumerate(frame_files, start=1):
                    src_path = os.path.join('Temp_frames', filename)
                # Put every 8 frames in output_main, every 9th frame in output_special, repeat pattern
                    if idx % 9 == 0:
                        dst_path = os.path.join(output_folder_Test, filename)
                    else:
                        dst_path = os.path.join(output_folder_Train, filename)
                shutil.move(src_path, dst_path)
                # Clean up temp folder
                os.rmdir('Temp_frames')
                window = create_test_window()
            elif event == 'Cancel':
                window.close()
                window = create_test_window()
        elif event == 'Import':
            window.close()
            window = Import_window()
            event, values = window.read()
            if event == 'Back':
                window.close()
                # Placeholder for actual import logic
                print("Import functionality would be executed here.")
                window = create_test_window()
        elif event == 'Export':
            window.close()
            window = Export_window()
            event, values = window.read()
            if event == 'Back':
                window.close() 
                # Placeholder for actual export logic
                print("Export functionality would be executed here.")
                window = create_test_window()
    window.close()
    
if __name__ == '__main__':
    main()