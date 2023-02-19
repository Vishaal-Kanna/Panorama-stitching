# Project 1 - AutoPano

  ## Dependencies to run both the phases
  1. Python 3.6 should be installed on your system.
  2. numpy - Install it using `python3 -m pip install numpy`
  3. matplotlib - Install it using `python3 -m pip install matplotlib`
  4. cv2 - Install it using `pip install opencv-python`
  5. os 
  6. argparse

  ## Instructions to run the Phase 1
  1. Since you are looking at this, We'll assume you unzipped YourDirectoryID_p1.zip
  2. Open command prompt or terminal.
  3. Navigate to this directory using `cd <your_path>/YourDirectoryID_p1/Phase1/Data/`
  4. Please make sure you load all images named ex: "1.jpg" in their respective datsets.
  5. Step 4 is essential for the code to run smoothly!!!
  6. Navigate to this directory using `cd <your_path>/YourDirectoryID_p1/Phase1/Code/`
  9. Run `python3 Wrapper.py --datatype <Train or Test> --dataset <ex: Set1>`
  10. Results will be displayed as well as saved in `cd <your_path>/YourDirectoryID_p1/Phase1/Results/`
    
  ## Instructions to run the Phase 2
  1. Since you are looking at this, We'll assume you unzipped YourDirectoryID_p1.zip
  2. Open command prompt or terminal.
  3. Navigate to this directory using `cd <your_path>/YourDirectoryID_p1/Phase2/Data/`
  4. Please make sure you load Train, Val, Test images into the appropriate folders here.
  5. Add suitable paths to the Training, Validation and Test folders if necessary in Train.py and Test.py (The relative path is given based on the file structure mentioned in the submission guidelines.)
  6. Navigate to this directory using `cd <your_path>/YourDirectoryID_p1/Phase2/Code/`
  9. Run `python3 Train.py --ModelType Supervised` (or Unsupervised, the default is set to run Supervised model) 
  10. The model weights are stored as weights.best.hdf5 in Checkpoints folder inside Phase2
  11. Change the path to the trained model in Test.py if necessary (the defaults is set to use weights.best.hdf5 that will be generated during training)
  12. Run `python3 Test.py --ModelType Supervised` (or Unsupervised, the default is set to run Supervised model)

 


