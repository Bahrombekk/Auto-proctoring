import subprocess

input_file = "Vakhid-Cs2-Anubis.mp4"
output_file = "trimmed_video.avi"
start_trim = 10
duration = 20

subprocess.run([
    'ffmpeg',
    '-ss', str(start_trim),
    '-i', input_file,
    '-t', str(duration),
    '-c', 'copy',
    '-avoid_negative_ts', 'make_zero',
    output_file
])

print("Eng tez ishlovchi kesma saqlandi:", output_file)
