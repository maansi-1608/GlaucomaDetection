# GlaucomaDetection
* This is a simple program to automate Glaucoma Diagnosis based on the cup and disk ratio of the retina. 
* OpenCV is primarily used for processing Glaucoma images and obtaining the radii of cup and disk. 
* A ratio >0.5 indicates the onset of Glaucoma. A ratio in the range of 0.7 - 0.9 indicates an advanced case of Glaucoma
* This program uses Kmeans clustering and Nerve Tracking approaces to isolate the retina in the images and then the red channel and green channel images are used to get the circumference of cup and disk to get their radius. 
