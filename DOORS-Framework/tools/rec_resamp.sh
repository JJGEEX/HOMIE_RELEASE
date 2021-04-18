echo "recording for 10 seconds"
arecord -t wav -D "hw:2,0" -d 10 -f S16_LE -r 48000 test.wav
# The all-important ffmpeg line
#ffmpeg -i test.wav -isr 48000 -ar 16000 downSamp.wav
ffmpeg -i test.wav -isr 48000 -ar 8000 downSamp.wav
echo "done recording"
#pocketsphinx_continuous -infile downSamp.wav > result.txt
#less result.txt
