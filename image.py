#from tensorflow import convert_to_tensor
from PIL import Image
import cv2
import numpy as np
from umeyama import umeyama
import shutil
import os
import glob
import platform


class Image_manipulation:

    def rotation_matrix(self, image, rotation_range=10, zoom_range=0.05, shift_range=0.05):
        h,w = image.shape[0:2]
        print(h, w)
        rotation = np.random.uniform( -rotation_range, rotation_range )
        scale = np.random.uniform( 1 - zoom_range, 1 + zoom_range )
        tx = np.random.uniform( -shift_range, shift_range ) * w
        ty = np.random.uniform( -shift_range, shift_range ) * h
        mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale )
        print(mat)
        mat[:,2] += (tx,ty)
        print(f"new mat: {mat}")
        result = cv2.warpAffine( image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE )
        if np.random.random() < 0.4:
            result = result[:,::-1]
        print(f"Rotation matrix value of result: {np.array(result).shape}\n")
        return result

       #'''This functions creates a rotation matrix by using the height and width
       #of the input image.'''
       #'''Define the values for the angle of rotation and the resizing'''
       #'''Angle of rotation and resizing value is generated through the interval defined below'''
       #angles = np.random.uniform(-rotation, rotation)
       #resize = np.random.uniform(1 - resizing, 1 + resizing)
       #'''Using the uniform distribution it takes in the highest - lowest value 
       #in the interval [a, b), where a - low and b- high. It includes the lowest value
       # in the distribution to generate a number but does not include the highest value'''
       #'''Getting the centre of the image which is going be used for the vector translation
       #added to the 2 dimensional rotation matrix'''
       #img = images
       #
       #h, w = img.shape[:2]
       #cen = (w / 2, h / 2)
       #'''Using cv2.getRotationMatrix2D to put the values into a rotational matrix'''
       #rotate_matrix = cv2.getRotationMatrix2D(cen, angles, resize)
       #'''To get the vector values for the translation to add to the 2x2 matrix to 
       # make it a 2x3 rotational matrix'''
       #y_vector = np.random.uniform(-vector, vector) * h
       #x_vector = np.random.uniform(-vector, vector) * w
       #trans_vector = (x_vector, y_vector)
       #'''X vector adds the first values in the first column and the y vector adds 
       #to the values in the second column'''
       #rotate_matrix[:, 2] += trans_vector
       #total = cv2.warpAffine(img, rotate_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
       #'''Getting the rotation on the image '''
       #if np.random.random() < 0.4:
       #    total = total[:,::-1]
       #return np.array(total)

    def warping_image(self, images):

        assert images.shape == (256,256,3)
        # range_ = numpy.linspace( 128-80, 128+80, 5 )
        # from center to border, how wide
        range_ = np.linspace( 128-128, 128+128, 5 )
        mapx = np.broadcast_to( range_, (5,5) )
        mapy = mapx.T

        mapx = mapx + np.random.normal( size=(5,5), scale=5 )
        mapy = mapy + np.random.normal( size=(5,5), scale=5 )

        interp_mapx = cv2.resize( mapx, (80,80) )[8:72,8:72].astype('float32')
        interp_mapy = cv2.resize( mapy, (80,80) )[8:72,8:72].astype('float32')

        warped_image = cv2.remap( images, interp_mapx, interp_mapy, cv2.INTER_LINEAR )

        src_points = np.stack( [ mapx.ravel(), mapy.ravel() ], axis=-1 )
        dst_points = np.mgrid[0:65:16,0:65:16].T.reshape(-1,2)
        mat = umeyama( src_points, dst_points, True )[0:2]

        target_image = cv2.warpAffine( images, mat, (64,64) )

        return warped_image, target_image

    def get_transpose_axes(self, n):
        if n % 2 == 0:
            y_axes = list( range( 1, n-1, 2 ) )
            x_axes = list( range( 0, n-1, 2 ) )
        else:
            y_axes = list( range( 0, n-1, 2 ) )
            x_axes = list( range( 1, n-1, 2 ) )
        return y_axes, x_axes, [n-1]

    def stack_images(self, images):
        images_shape = np.array( images.shape )
        new_axes = self.get_transpose_axes( len( images_shape ) )
        new_shape = [ np.prod( images_shape[x] ) for x in new_axes ]
        return np.transpose( images, axes = np.concatenate( new_axes )).reshape( new_shape )


    def image_stack(self, images):
        imgs = np.array(images.shape)
        img = np.ones(imgs)
        print(img)
        print(type(img))
        n = len(img)
        #img = np.ones(imgs)
        val_1 = list(range(1, n - 1, 2))
        val_2 = list(range(0, n - 1, 2))
        print(img)
        if n % 2 == 0:
            axes = (val_1, val_2, [n - 1])
        else:
            axes = (val_2, val_1, [n - 1])
        print(f"values of axes is: {axes}\n")
        print(f"length of axes is: {len(axes)}\n")
        print(type(axes))
        if type(axes) == tuple:
            transposed_axes = np.transpose(imgs, axes=np.concatenate(axes))
            print(type(transposed_axes))
            new_arr_shape = []
            for x in axes:
                new_arr_shape.append(np.prod(img[x]))
            print(type(new_arr_shape))
            print(new_arr_shape)

            for index, x in enumerate(new_arr_shape):
                new_arr_shape[index] = int(x)

            #print(new_arr_shape)
            #print(len(transposed_axes))

            final_shape = transposed_axes.reshape(new_arr_shape)
            print(type(final_shape))
            #print(final_shape)



class Training_Utilities(Image_manipulation):

    def __init__(self):
        super().__init__()

    def training_data(self, images, batch_size):
        # i = np.random.randint(0, high=len(imgs), size=batch_size)
        '''Creating a batch '''

        # for k in i:
        #     img = imgs[k]

        #     rotated_img = self.rotation_matrix(image=img)

        #     warped_img = np.array(self.warping_image(images=rotated_img))
        #     target_img = convert_to_tensor(self.warping_image(images=rotated_img))
            
        #     print(type(warped_img), type(target_img))
        #     if k == 0:
        #         warped_imgs = np.empty(((batch_size,) + warped_img.shape, warped_img.dtype))
        #         target_imgs = np.empty(((batch_size,) + target_img.shape, target_img.dtype))
        #         warped_imgs[k] = warped_img
        #         target_imgs[k] = target_img
            
        #     return target_img, warped_img

        #indices = np.random.randint( len(imgs), size=batch_size )
        #for i, index in enumerate(indices):
            #image = imgs[index]
            #images = self.rotation_matrix(image)
           # warped_img, target_img = self.warping_image(image)

           # if i == 0:
               #warped_images = np.empty( (batch_size,) + warped_img.shape, warped_img.dtype )
              # target_images = np.empty( (batch_size,) + target_img.shape, warped_img.dtype )

           # warped_images[i] = warped_img
           # target_images[i] = target_img

        # return warped_images, target_images
        indices = np.random.randint( len(images), size=batch_size )
        for i,index in enumerate(indices):
            image = images[index]
            image = self.rotation_matrix( image)
            warped_img, target_img = self.warping_image( image )

            if i == 0:
                warped_images = np.empty( (batch_size,) + warped_img.shape, warped_img.dtype )
                target_images = np.empty( (batch_size,) + target_img.shape, warped_img.dtype )

            warped_images[i] = warped_img
            target_images[i] = target_img

        return warped_images, target_images
       




class Video:
    def converted_images(self, ae, images):
        print(images)
        i = 0
        for imgs in images:
            val = cv2.imread(imgs)
            #print(val.shape)
            crop = slice(48, 208)
            print(f'crop variable class type{type(crop)}')
            face = val[crop, crop] 
            print('Face crop completed')
            face = np.expand_dims((cv2.resize(face, (64, 64))), 0)
            print('Expanding Dimensions complete')
            print(f"face.shape: {face.shape}")
            print(type(face))
            gen_face = ae.predict(face/255.0)[0]
            print('prediction done')
            gen_face = cv2.resize((np.clip(gen_face * 255, 0, 255).astype(val.dtype)), (160, 160))
            print('Gen face resize completed')
            new_img = val.copy()
            print('New img created ')
            new_img[crop, crop] = gen_face
            print('Crop values completed')
            cv2.imwrite((str(i) + '.jpg'), face)
            print('Creating...' + str(i) + '.jpg')
            i += 1

    # DONE
    def video_to_images(self, vid, folder):
        '''Use cv2 to capture the video'''
        capture = cv2.VideoCapture(vid)

        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except OSError:
            print(f'Error creating {folder}')

        i = 0
        video_time = 0
        fps = 1 / 10

        while True:
            video_time += fps
            capture.set(cv2.CAP_PROP_POS_MSEC, video_time * 1000)
            '''Read the video captured by cv2'''
            ret, frame = capture.read()
            if ret:
                name = folder + "\\" + str(i) + '.jpg'
                print('Creating...' + str(i) + '.jpg')
                '''Use cv2.imwrite to write the write the video 
                into frames and name the frames as well'''
                cv2.imwrite(name, frame)
                i += 1
            else:
                break
        '''Release the video after the frames have been created'''
        capture.release()
        '''To take it out of the loop otherwise it will infinitely create the frames'''
        cv2.destroyAllWindows()

    # DONE
    def loading_images(self, img_p, convert=None):
        image_paths = []
        for filename in glob.glob(img_p + '\\*.jpg'):
            #    '''Add it into the array'''
                 image_paths.append(filename)
        iter_all_images = ( cv2.imread(fn) for fn in image_paths )
        if convert:
           iter_all_images = ( convert(img) for img in iter_all_images )
        for i,image in enumerate( iter_all_images ):
           if i == 0:
              all_images = np.empty( ( len(image_paths), ) + image.shape, dtype=image.dtype )
        all_images[i] = image
        return all_images


        # 1: loading full path of images from directory to list
        # 2: Reading images using cv2.imread() 

        #images = []
        #if platform.system() == 'Windows':
            #for filename in glob.glob(img_p + '\\*.jpg'):
            #    '''Add it into the array'''
            #    images.append(filename)
      
        #image_file_name = os.listdir(str(img_p))
        #images = []
        #for image in image_file_name:
           # images.append(img_p + str(image))

        #print(images)
        #print('Current total of images in array... ' + str(len(images)))
        
       #elif platform.system() == 'Linux':
       #    for filename in glob.glob(img_p + '/*.jpg'):
       #        '''Add it into the array'''
       #        images.append(filename)
       #        print('Current total of images in array... ' + str(len(images)))

        #read_imgs = []
        #for img in images:
           # read_imgs.append(cv2.imread(img, 1))
        #read_imgs = (cv2.imread(imgs) for imgs in images)
        #print(read_imgs)
        #print(read_imgs)
        #return read_imgs

       #for i, img in enumerate(read_imgs):
       #    if i == 0:
       #        all_imgs = np.empty((len(images), ) + img.shape, dtype=img.dtype)
       #    all_imgs[i] = img
       #    return all_imgs


    # DONE
    def move_file(self, name, images, tar, origin):
        if name == 'video':
            shutil.move(origin, tar)
        elif name == 'image':
            img_array = []
            file_array = []
            '''Globbing to only get the files in the folder that end with .jpg'''
            for filename in glob.glob("resized_" + images + '\\*.jpg'):
                '''Add it into the array'''
                file_array.append(filename)
            file = int(len(file_array))
            print(file)
            '''Check if the number is of the required amount otherwise it won't transfer'''
            if file == len(file_array):
                for i in range(0, file):
                    img_array.append(str(i) + '.jpg')
                    print(len(img_array))
                '''Move the images into the target folder by using a loop'''
                for j in range(len(img_array)):
                    images = origin + '\\' + str(img_array[j])
                    shutil.move(images, tar)
                    print('Shifting.... ' + str(img_array[j]))
            else:
                pass
        else:
            pass

    # DONE
    def get_file_path(self, file):
        if platform.system() == "Linux":
            return os.getcwd() + '/' + file + '/'

        if platform.system() == "Windows":
            return os.getcwd() + '\\' + file + '\\'
        # for dirpath, dir, files in os.walk(dir):
        #     '''Walk through the OS directories to find the specified directory'''
        #     if file in files:
        #         '''If the file is in the specified files within the specified dir
        #         then return the path of the directory + file'''
        #         '''where dir = "C:\\Users\\user\\" and file = "video_name.mp4"'''
        #         return os.path.join(dirpath, file)


    def resize_images(self, ori_folder, tar_folder):
        
        images = os.listdir(ori_folder)
        image_paths = []
        for image in images:
            image_paths.append(ori_folder + '\\' + image)

        image_name = 0
        for path in image_paths:
            basewidth = 256
            #basewidth = 64
            img = Image.open(path)
            img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
            img.save(os.getcwd() + '\\' + tar_folder  + '\\' + str(image_name) + '.jpg')
            print(f"Successfully saved image ... {str(image_name) + '.jpg'}")
            image_name += 1

    # DONE
    def make_dir(self, dir_name):
        '''Takes dir_name as path to directory which you need to find
        and checks if the directory exists, if it does not it creates
        the directory
        :param dir_name Name of directory'''
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        else:
            pass

    # DONE
    def convert_frame_to_video(self, directory):
        # Selecting path to the images folder depending upon your operating system
        if platform.system() == "Windows":
            image_files = os.listdir(os.getcwd() + "\\" + directory + "\\")
            image_directory = os.getcwd() + "\\" + directory + "\\"

        elif platform.system() == "Linux":
            image_files = os.listdir(os.getcwd() + "/" + directory + "/")
            image_directory = os.getcwd() + "/" + directory + "/"

        # Making the images in the list to be in the correct order
        numbered_images = []
        for image in image_files:
            numbered_images.append(int(image.rstrip('.jpg')))

        for i in range(len(numbered_images)):
            for y in range(i + 1, len(numbered_images)):
                if numbered_images[i] > numbered_images[y]:
                    z = numbered_images[i]
                    numbered_images[i] = numbered_images[y]
                    numbered_images[y] = z

        image_files = []
        for x in numbered_images:
            image_files.append(str(x) + '.jpg')

        # Converting images into 3D matrix (since pictures are coloured) using cv2.imread() -> I will be referring to these as 'read-images'
        processed_images_list = []

        for image_index, image in enumerate(image_files):
            path = image_directory + image_files[image_index]
            print(f"Creating read image...{image}")
            processed_image = cv2.imread(path, 1)
            height, width = processed_image.shape[0], processed_image.shape[1]
            processed_images_list.append(processed_image)

        # Creating a cv2.VideoWriter() object in order to convert the read images into a video
        video_name = "finished_video.avi"
        videowriter_object = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc("D", "I", "V", "X"), 10,
                                             (width, height))

        # Making a video using the read images through the cv2.VideoWriter.write() method
        x = 1
        for image in processed_images_list:
            if x == 1:
                print("\nGenerating video with read images...")
            x += 1
            videowriter_object.write(image)

        print(f"Sucessfully generated {video_name}!")

        # Releasing the cv2.VideoWriter() object
        videowriter_object.release()





