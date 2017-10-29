import tensorflow as tf
import argparse
import numpy as np
from scipy import stats
from os import listdir, path, walk
from tqdm import tqdm
from skimage import io, transform

from segmentationGenerator import SegmentationGenerator

ROOT_FOLDER = './dataset/'
DESTINATION_ROOT = 'segmentedImages/'
SRC_ROOT = 'extracted/'
SUBFOLDER = 'humanshot/'

DATA_TXT = './data/seg/images'


def parseArguments():
    parser = argparse.ArgumentParser(description='Generate semantic segmentations with VGG FCN8')

    parser.add_argument('--create-txt', action='store_true', default=False, dest='createTxt',
                        help='True: if want to create txt file listing image paths to perform segmentation')
    parser.add_argument('--img-src', action='store', default='humanshot/others/', dest='imgSrc',
                        help='Path to the txt file listing images to perform segmentation')
    parser.add_argument('--batch-size', action='store', default=1, dest='batch_size',
                        help='Size of the batch')
    parser.add_argument('--read-npy', action='store_true', default=False, dest='readNpy',
                        help='True: Read numpy file')
    parser.add_argument('--subfolder', action='store', default=SUBFOLDER, dest='subfolder',
                        help='360 or humanshot')
    return parser.parse_args()


def _readFilenames(imageListFile, folder):
    fileNames = []
    with open(imageListFile, 'r') as f:
        for line in tqdm(f, desc='Reading image file names at: ' + imageListFile):
            fileNames.append(folder + '/' + line.strip())
    return fileNames


def _createImagesTxt(args, folder):
    img_lists = sorted(listdir(ROOT_FOLDER + SRC_ROOT + args.subfolder + folder))
    newFile = DATA_TXT + '_' + folder + '.txt'
    with open(newFile, 'w') as file:
        for f_names in tqdm(img_lists, desc='Writing train.txt for input dataset'):
            if path.isdir(ROOT_FOLDER + SRC_ROOT + args.subfolder + folder+'/' + f_names):
                continue  # skip directories
            file.write(f_names + '\n')
    return newFile


def readImagesFromDisk(args, fileNames, reshape=None):
    images = []
    file_names = []
    for i in tqdm(range(0, len(fileNames)), desc='Reading images from disk'):
        img = io.imread(ROOT_FOLDER + SRC_ROOT + args.subfolder + fileNames[i])
        if reshape is not None and img.shape != reshape:
            img = transform.resize(img, reshape, mode='reflect', preserve_range=True)
        # img = tf.read_file(fileNames[i])
        # img = tf.image.decode_jpeg(img, channels=3)
        # if reshape is not None and img.shape != reshape:
        #     img = tf.image.resize_images(img, reshape)
        images.append(img)
        file_names.append(fileNames[i])
    return np.array(images), file_names


def saveImages(seg_images, file_names):
    segmentedImages = []
    folder_path = ROOT_FOLDER + DESTINATION_ROOT + args.subfolder
    foldername = ''
    for j in range(0, len(seg_images)):
        imageName = file_names[j]
        imagePath = ROOT_FOLDER + DESTINATION_ROOT + args.subfolder + imageName
        foldername = imageName.split('/')[0]

        def map_channels(i_x):
            i, x = i_x
            x = (x * 255).astype(np.uint8)
            if x.max() > 0.35 * 255:
                threshold = np.fabs(x.max() - x.max() * .65)
            else:
                threshold = 255
            threshImage = stats.threshold(x, threshmin=threshold)
            threshImage[threshImage > 0] = i
            return threshImage

        def smash_channels(channels):
            base = channels[0]
            for i, x in enumerate(channels):
                base[x > 0] = i
            return base

        imgchannels = list(map(map_channels, enumerate(np.transpose(seg_images[j, :, :, :], [2, 0, 1]))))
        smashed = smash_channels(imgchannels)
        segmentedImages.append(smashed)
        #io.imsave(imagePath, smashed)
    np.savez_compressed(folder_path + foldername + '_segmented_frames', seg_frames=segmentedImages)


def generateSegmentation(images_list, file_names, batch_size):
    total = images_list.shape[0]
    height = images_list.shape[1]
    width = images_list.shape[2]
    channels = images_list.shape[3]
    batch_shape = [batch_size, height, width, channels]

    batch_iterations = total // batch_size
    leftovers = total % batch_size

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    tf.reset_default_graph()
    #with tf.Session(config=config) as sess:

    segmentationGen = SegmentationGenerator(input_shape=batch_shape)#, sess=sess)

    segmented_images = np.array([])
    for itr in range(0, batch_iterations * batch_size, batch_size):
        batch_images = images_list[itr:itr + batch_size]
        # batch_filenames = file_names[itr:itr + batch_size]
        if itr == 0:
            segmented_images = segmentationGen.getSegmentation(batch_images, smashed=False)
        else:
            segmented_images = np.concatenate([segmented_images, segmentationGen.getSegmentation(batch_images, smashed=False)],
                                              axis=0)
        print('batch processed '+ str(itr))

    # leftovers
    if leftovers > 0:
        batch_images = images_list[-leftovers:]
        # batch_filenames = file_names[-leftovers:]
        segmented_images = np.concatenate([segmented_images, segmentationGen.getSegmentation(batch_images, smashed=False)],
                                          axis=0)
        # saveImages(segmented_images, batch_filenames)
        print('leftover batch processed')

    saveImages(segmented_images, file_names)


if __name__ == '__main__':
    args = parseArguments()
    print(args)

    folders = [d for d in listdir(ROOT_FOLDER + SRC_ROOT + args.subfolder) if
               path.isdir(path.join(ROOT_FOLDER + SRC_ROOT + args.subfolder, d))]

    for fol in folders:
        if args.readNpy:
            loaded = np.load(ROOT_FOLDER + SRC_ROOT + args.subfolder + fol + '/npy/frames.npz')
            images_list = loaded['frames']
            file_names = loaded['frame_names']
        else:
            if args.createTxt:
                _createImagesTxt(args, fol)
            #filenames = _readFilenames(DATA_TXT + '_' + fol + '.txt', fol)
            images_list, file_names = readImagesFromDisk(args, ['New Folder/test.jpg','New Folder/test2.JPG','New Folder/test3.jpg'], [400, 684, 3])

        generateSegmentation(images_list, file_names, args.batch_size)
