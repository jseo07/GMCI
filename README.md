# GMCI
Produces artificial images of human faces using EITHER StyleGAN and PGAN with the produced images gradually increasing in 'resemblance' to the user's intentions based on user inputs.

This is a program that intends to produce an image of a human face that a user has in mind using the already existing, pre-trained StyleGAN or PGAN. For instance, if you wish to produce an image of a face with the criteria of "dark skinned, middle aged, man with beard", the program attempts its best to produce an image that best satisfies the given criteria. 

Instructions for testing:
1. Clone gmci.py and run.
2. When images load, consider the 8 images from the right, ignoring the seed image and one image under it; they are unnoised seed image, not intended to be rated.
3. Have a certain facial features in mind, such as the skin colour, gender, age, etc. 
4. Rate the image on a scale of 0 to 10, in terms of their resemblence to the facial features in mind, i.e, give an image that matches the features you have in 
mind a higher rating and vice versa. Rate all 8 images, excluding the seed images described in number 1.
5. Go over the cycle until the images approach the facial features you had in mind.

There are 4 functions that utilizes the user inputs to produce an appropriate noise for the GAN. You may test the different functions by uncommenting the call for the function to be tested and commenting the call for the function that was previously in use, at the end of gmci.py.


