# jubeatvisualize

_(Work In Progress)_

To use this visualization tool, the relevant train .npy files into the train/ folder for the HMM model. The train/ folder directory should look like this: 

```
train/
-- data/
---- (.npy data files here)
-- label/
---- (.npy label files here)

```

You should also place your trained CNN model file, and edit the variable CNNMODELFILEPATH in the main.py file. 

Once everything is in place, you can run this Desktop application by running these commands from the root directory:

```
cd jubeatvisualize/
python3 main.py 
```

*Note: Do make sure that your computer is configured to be compatible to run PyQt5.