# captury.py
import subprocess
import time

input_file = "Vakhid-Cs2-Anubis.mp4"
output_file = 'full_record.avi'
trimmed_output = 'trimmed_video.avi'

subprocess.run([
    'ffmpeg', '-i', input_file,
    '-c:v', 'libxvid', '-c:a', 'mp3',
    output_file
])

start_trim = 10
duration = 20

subprocess.run([
    'ffmpeg', '-i', output_file,
    '-ss', str(start_trim), '-t', str(duration),
    '-c:v', 'copy', '-c:a', 'copy',
    trimmed_output
])

print("Kesilgan video saqlandi:", trimmed_output)