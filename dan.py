'''
    Need to create sequences for both '.' and '_'

    def createDot(directory)
    def createDash(directory)
        where the directory passed is the one containing 4 frames 
        for the creation (labeled closed.jpg, open.jpg, opcl.jpg, clop) of the Dot and Dash.
***** Please make sure theyre taken from consecutive frames so as to not have much movement from face ******

    def create nums(directory)
        where the create dot and dash are in
    Numbers in morse code:
        '0':'-----',        '1':'.----',         '2':'..---',
        '3':'...--',        '4':'....-',         '5':'.....',
        '6':'-....',        '7':'--...',         '8':'---..',
        '9':'----.', 
    
    If we ever get to it alphabet in morse code:
        'A':'.-',           'B':'-...',          'C':'-.-.',
        'D':'-..',          'E':'.',             'F':'..-.',
        'G':'--.',          'H':'....',          'I':'..',
        'J':'.---',         'K':'-.-',           'L':'.-..',
        'M':'--',           'N':'-.',            'O':'---',
        'P':'.--.',         'Q':'--.-',          'R':'.-.',
        'S':'...',          'T':'-',             'U':'..-', 
        'V':'...-',         'W':'.--',           'X':'-..-', 
        'Y':'-.--',         'Z':'--..',
'''
import cv2 
import os
import sys
from nami import *

# Create the Dot transmutation calls
def createDot(openF, closeF, opclF, clopF, idx):
    # Get all the information to be able to concat images into video
    height, width, layers = openF.shape
    # Message if succesful
    message = 'called createDot'
    # Create vid to be returned
    try:
        frameArr = []
        frameArr.append(openF)
        frameArr.append(opclF)
        frameArr.append(closeF)
        frameArr.append(clopF)
        frameArr.append(openF)
        frameArr.append(openF)
        frameArr.append(openF)
        CatFrames(frameArr, 'p%d_dot'%idx, 'PD')
        
        # Print if successful
        print(message)
        # Close everything
        cv2.destroyAllWindows()
        dotVid.release()
        # return the video
        return None
    except:
        print("Oops!",sys.exc_info()[0],"occured.")

# Create the Dash transmutation
def createDash(openF, closeF, opclF, clopF, idx):
    # Get all the information to be able to concat images into video
    height, width, layers = openF.shape
    message = 'called createDash'
    try:
        frameArr = []
        frameArr.append(openF)
        frameArr.append(opclF)
        frameArr.append(closeF)
        frameArr.append(closeF)
        frameArr.append(closeF)
        frameArr.append(clopF)
        frameArr.append(openF)
        CatFrames(frameArr, 'p%d_dash'%idx, 'PD')

        # Print if successful
        print(message)
        # Close everything
        cv2.destroyAllWindows()
        dashVid.release()
        # return the video
        return None
    except:
        print("Oops!",sys.exc_info()[0],"occured.")

# Create all the transmutations of 0-9 from new directory created from createDot/createDash
# Doesnt get called until both createDot and CreateDash are finished
def createNums(directory):
    original_imgs = []
    # create array containing only original images to be augmented 
    for x in os.walk(directory):
        original_imgs.append(os.path.abspath(x[0]))
    # remove ./Frames from array
    del original_imgs[0]
    # use data_aug per folderto create all data
    for img in original_imgs:
        data_augment(img)
    
    #data_augment(directory)
    all_paths = []
    for x in os.walk(directory):
        all_paths.append(os.path.abspath(x[0]))
        print(os.path.abspath(x[0]))
    del all_paths[0]
    # only do something if checkDir passes
    for i in range(0, len(all_paths)):
        abs_path = all_paths[i]
        if checkDir(abs_path):
            # Store in variables to later recall
            openF  = cv2.imread('open.jpg')
            closeF = cv2.imread('close.jpg')
            opclF  = cv2.imread('opcl.jpg')
            clopF  = cv2.imread('clop.jpg')
            # Call createDot and createDash to create them from ^
            createDot(openF, closeF, opclF, clopF, i)
            createDash(openF, closeF, opclF, clopF, i)
            os.chdir('../')
            # Create all the numbers if ^ was succesful
            print('videos created')





# Checks to make sure all the files are included in the passed directory
def checkDir(directory):
    # Move to cur directory
    if not os.path.exists(directory):
        print('Path doesnt exist')
        return False
    print(directory)
    os.chdir(directory)
    # Store values to check
    openExist = os.path.exists('open.jpg')
    closeExist = os.path.exists('close.jpg')
    opclExist = os.path.exists('opcl.jpg')
    clopExist = os.path.exists('clop.jpg') 
    
    # Check if they exist
    if openExist and closeExist and opclExist and clopExist:
        print('All the files exist')
        return True
    elif not openExist and closeExist and opclExist and clopExist:
        print('Missing open.jpg')
        return False
    elif openExist and not closeExist and opclExist and clopExist:
        print('Missing close.jpg')
        return False
    elif openExist and closeExist and not opclExist and clopExist:
        print('Missing opcl.jpg')
        return False
    elif openExist and closeExist and opclExist and not clopExist:
        print('Missing clop.jpg')
        return False
    # If multiple dont exist
    else:
        print('Missing multiple files')
        return False



# Concatenates frames into a single video file.
#   frames[array of im] = array of images (size 4 for out dataset).
#   identifier[string] = output video file name that identifies sequence.
#   person[string] = which person does this data belong to (PT, PD, PN).
def CatFrames(frames, identifier, person):
    output_path = '\Dataset\%s\RawData\%s.avi' % (person, identifier)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = '%s%s' % (dir_path, output_path)
    print(dir_path)
    output_path = dir_path
    # determine width and height of image.
    height, width, channels = frames[0].shape

    # define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()