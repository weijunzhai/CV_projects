from PIL import Image
import numpy as np
import os

class Eigenfaces:
    labels = []               # list to store label names of the training images
    U = []                    # list to store eigenVectors of AA.T
    Omiga = []                # weights of taining images
    meanImageArray = []       # the mean value list of the training images

    def __init__(self):
        self.U = []
        self.Omiga = []
        self.labels = []
        self.meanImageArray = []

    # this method will train the input images and calculate the value we want for the face recognize process
    # the input imageFolder is the folder source which contains training images
    def training(self, imageFolder):
        # loading the training image and convert those images into matrix values
        imagesArray, labels = self.load_images(imageFolder)
        # initialize the labels value to the object
        self.labels = labels
        # generate an mean image array
        self.meanImageArray = imagesArray.mean(1)
        # calculating the difference of the training array to their meanimage vector
        A = self.diff(imagesArray, self.meanImageArray)
        arr = np.dot(A.T, A)
        # calculate out the eigenVectors and the eigenValues
        eigenValues, eigenVectors = self.compute_eigenValues_eigenVectors(arr)
        self.U = np.dot(A, eigenVectors)
        norms = np.linalg.norm(self.U, axis=0)
        # normalize the eigenVectors
        self.U = self.U / norms
        # initialize weights of the training images
        self.Omiga = np.dot(self.U.T, A)

    # method uses to load the images from the folder: imageFolder
    # this method will return the images's gray value as a matrix with each image as a vector
    # the other return list labels stores the image's filename
    def load_images(self, imageFolder):
        images = os.listdir(imageFolder)
        labels = images
        data = np.empty((len(images), 195 * 231), dtype="float64")
        for i in range(len(images)):
            image = Image.open(imageFolder + "/" + images[i])
            arr = np.asarray(image, dtype="float64")
            arr = arr.flatten()
            data[i, :] = arr

        return data.T, labels

    # method diff: calculate the differece between the input image matrix and the meanMatrix of the training images
    def diff(self, matrix, meanArray):
        matrix = matrix.T
        matrix = [arr - meanArray for arr in matrix]
        matrix = np.array(matrix)
        return matrix.T

    # this method will generate a mean image based on the input image matrix
    def generate_mean_image(self,):
        meanImage = np.array(self.meanImageArray)
        meanImage = meanImage.reshape((231, 195))
        meanImage = Image.fromarray(meanImage)
        meanImage = meanImage.convert("RGB")
        meanImage.save("meanImage.jpg", "jpeg")

    # this method comput the eigenVectors and the eigenValues
    def compute_eigenValues_eigenVectors(self, arr):
        # compute out the eigenValues and the eigenVectors by using the method provided by numpy
        eigenValues, eigenVectors = np.linalg.eig(arr)
        # sort the vector
        idx = np.argsort(-eigenValues)
        eigenValues = eigenValues[idx]
        # sort the eigenvectors based on the eigenvlues
        eigenVectors = eigenVectors[:, idx]
        eigenVectors = eigenVectors.T
        # rule out eigenvectors which has less eigenvalues
        eigenVectors = eigenVectors[0:len(eigenVectors)-3]
        return np.array(eigenValues), eigenVectors.T

    # save all the eigenface needed for project 2
    def eigenfaceToImages(self):
        faces = self.U.T
        if not os.path.exists("./eigenface/"):
            os.mkdir("./eigenface/")
        for i in range(len(faces)):
            face = faces[i]
            n = np.min(face)
            m = np.max(face)
            temp = [((i-n)*255.0)/(m-n) for i in face]
            temp = np.array(temp, dtype="int32")
            temp = np.reshape(temp, (231, 195))
            img = Image.fromarray(temp)
            img = img.convert('RGB')
            imageName = self.labels[i]
            img.save("./eigenface/"+imageName, "jpeg")

    # save all the testSubtractMeanImages images needed for project 2
    def testSubtractMeanImages(self, test_A, testImages):
        faces = test_A.T
        if not os.path.exists("./testImageAfterSubtractMean/"):
            os.mkdir("./testImageAfterSubtractMean/")
        for i in range(len(faces)):
            face = faces[i]
            n = np.min(face)
            m = np.max(face)
            temp = [((i - n) * 255.0) / (m - n) for i in face]
            temp = np.array(temp, dtype="int32")
            temp = np.reshape(temp, (231, 195))
            img = Image.fromarray(temp)
            img = img.convert('RGB')
            imageName = "./testImageAfterSubtractMean/" + testImages[i]
            img.save(imageName, "jpeg")

    # save all the RECONSTRUCT FACE images needed for project 2
    def reconstructFaces(self, reconstructe_I, testImages):
        faces = reconstructe_I.T
        if not os.path.exists("./recontructedFace/"):
            os.mkdir("./recontructedFace/")
        for i in range(len(faces)):
            face = faces[i]
            n = np.min(face)
            m = np.max(face)
            temp = [((i - n) * 255.0) / (m - n) for i in face]
            temp = np.array(temp, dtype="int32")
            temp = np.reshape(temp, (231, 195))
            img = Image.fromarray(temp)
            img = img.convert('RGB')
            imageName = "I_" + testImages[i]
            img.save("./recontructedFace/"+imageName, "jpeg")

    # compute the vector distance of two matrices
    def vector_dist(self, matrix1, matrix2):
        length = len(matrix1[0],)
        matrix1 = matrix1.T
        matrix2 = matrix2.T
        dist = []
        for i in range(length):
            dist.append(self.distance(matrix1[i], matrix2[i]))
        dist = np.array(dist)
        return dist

    # compute distance between two vectors
    def distance(self, vector1, vector2):
        return np.linalg.norm(vector1-vector2)

    # based on the test matrix we get from the test images. we will find our the vector that is cloest to the test image
    # return the list with the training labels and their distance to test images
    def find_dj_and_label(self, test_matrix):
        test_matrix = test_matrix.T
        train_matrix = self.Omiga.T
        labels = []
        distance = []
        for i in range(len(test_matrix)):
            min_num = 3.24094697e+50
            min_j = 0
            for j in range(len(train_matrix)):
                dist = self.distance(test_matrix[i], train_matrix[j])
                if dist < min_num:
                    min_num = dist
                    min_j = j
            distance.append(min_num)
            labels.append(self.labels[min_j])
        return labels, np.array(distance)

    # identified face in the test folder, then point out the labels of the test image.
    # testFolder: folder contains test images
    def predict(self, testFolder, T0, T1):
        # transfer test images to the test array according to their gray values
        testArray, testImages = self.load_images(testFolder)

        # (1) set T0 to 13900, set T1 to 8500
        print("(1) the chosen thresholds T0 and T1")
        print("T0 =", T0, "T1 =", T1)

        # (2) output the mean face of the training images
        print("")
        print("(2) output the mean face of the training images")
        self.generate_mean_image()
        # output the eigenface of the training images
        self.eigenfaceToImages()

        # (3) Print out the PCA coefficients for each training image
        print("")
        print("(3), PCA coefficients for each training image")
        for i in range(len(self.Omiga.T)):
            print(self.labels[i], [float("%.2f" % j) for j in self.Omiga.T[i]])

        # (4) Output all the test images after subtracting the mean face.
        print("")
        print("(4) Output all the test images after subtracting the mean face")
        # compute the diff matrix of the test matrix
        test_A = self.diff(testArray, self.meanImageArray)
        self.testSubtractMeanImages(test_A, testImages)

        # (5) Print out test images PCA coefficients
        print("")
        print("(5) PCA coefficients for each test image")
        Omiga_I = np.dot(self.U.T, test_A)
        for i in range(len(Omiga_I.T)):
            print(testImages[i], [float("%.2f"%j) for j in Omiga_I.T[i]])

        # (6) Output the reconstructed face image
        # reconstruct the input vectors
        reconstruct_I = np.dot(self.U, Omiga_I)
        print("")
        print("(6) Output the reconstructed face image")
        self.reconstructFaces(reconstruct_I, testImages)

        # compute out the distance between the input and the reconstruct vectors
        d0 = self.vector_dist(reconstruct_I, test_A)
        # for every test image, find out the images in the training set which has least distance
        result, distance = self.find_dj_and_label(Omiga_I)

        # (7)	Print out the distance di for I =0 to M
        print("")
        print("(7) distance di for I =0 to M")
        for i in range(len(result)):
            print(i, testImages[i], float("%.2f"%distance[i]))

        # (8) Print out the class result
        print("")
        print("(8) class result")
        for i in range(len(result)):
            # if the distance is larger than T0, then this image does not contains a face
            if d0[i] > T0:
                print(testImages[i], "is not a face image")
            # if the distance is larger than T1, then this image does not have identical face in the training set
            elif distance[i] > T1:
                print(testImages[i], "is a unknown face")
            # if the distance is smaller than T1, then find the image set with the same face in the training
            else:
                print(testImages[i], "has the same face with", result[i])


if __name__ == '__main__':
    # create an object of Eigenfaces
    eigenFace = Eigenfaces()
    # put the training images in a folder, and call the training function
    eigenFace.training("./TrainingImage/")

    # (1) set T0 to 13900, set T1 to 8200
    T0 = 13900
    T1 = 8200

    # test test images
    eigenFace.predict("./Face dataset/", T0, T1)
