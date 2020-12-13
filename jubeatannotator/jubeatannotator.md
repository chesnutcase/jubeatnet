# jubeatannotator

_(Work In Progress)_

To use this annotator, put <song_name>.txt into the ```resources/``` folder as well as the corresponding video. Make sure that the .txt file is lower case, and the spaces are replaced with underscores. i.e. ```red_zone.txt```. 

You should also place an excel or csv file containing video information here. An example has been provided for you in ```resources/info.xlsx```.

Once the video you want to annotate and the matching beatmap are in place, you can run this Desktop application by running these commands from the root directory:

```
cd jubeatannotator/
python3 main.py 
```

Your labelled files will be saved in the ```out/``` folder. 

*Note: Do make sure that your computer is configured to be compatible to run PyQt5