MODEL="/modules/model"
CURDIR=`pwd`
if [ -d "$CURDIR$MODEL" ] 
then
	echo "$MODEL installed"
else
	echo "$MODEL not installed"
	echo "installing now"
	mkdir "$CURDIR$MODEL"
	wget https://alphacephei.com/vosk/models/vosk-model-en-us-daanzu-20200905.zip -P $CURDIR$MODEL #download model
	unzip $CURDIR$MODEL/vosk*
	mv vosk* vosk-model #easier rename for moving stuff
	mv vosk-model/* $CURDIR$MODEL #move stuff to correct directory
	rm -r vosk-model #rm temp directory
fi
pip3 install vosk #voice recognition
pip3 install sklearn #comparing spoken commans to known commands
pip3 install scikit-learn #comparing spoken commands to known commands
pip3 install fuzzywuzzy #music title comparison
pip3 install eyed3 #checking tags for music
pip3 install word2number #coverting words to ints
pip3 install python-Levenshtein #for checking how close commands commands are to each other
pip3 install flux_led #for interacting with the flux lightbulb
pip3 install youtube-search-python #searching for youtube music videos
pip3 install pygame #playing audio out back-end for testing
pip3 install parse #for parsing spoken commands
pip3 install youtube_dl #for downloading music MAYBE NOT NEEDED
pip3 install pafy #for finding youtube paths for music
pip3 install pydub #got silence detection
pip3 install mutagen #music tag detection
pip3 install tinytag #music tag detection
pip3 install python-kasa --pre #for hs103 smartplug
pip3 install pytz
sudo apt-get update
sudo apt-get install espeak
sudo apt-get install libgfortran5
sudo apt-get install libgfortran3
sudo apt-get install libatlas-base-dev
