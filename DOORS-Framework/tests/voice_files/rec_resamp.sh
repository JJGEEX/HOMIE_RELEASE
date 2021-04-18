echo "recording for 5 seconds"
arecord -t wav -D "hw:2,0" -d 5 -f S16_LE -r 48000 test.wav
# The all-important ffmpeg line
ffmpeg -i test.wav -isr 48000 -ar 8000 downSamp.wav
#ffmpeg -i test.wav -isr 48000 -ar 8000 downSamp.wav
echo "done recording"
