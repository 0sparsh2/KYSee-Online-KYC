# KYSee Online KYC

A 5 step online KYC portal that verifies your uploaded documents with your data using the applications of Machine Learning.
The verification steps include:

1)OTP generation and verification through SMTP server

2)Perosnal details veriifcation from Pancard by augmenting the uploaded image and extracting the text. (Used OpenCV and Pytesseract)

3)Address verification from Adhaar card/ Passport/ Driver's Liscene by the similar method

4)Signature verification by signing on a javascript whiteboard plugin and comparing with the one of Pancard using SSIM and MSE

5)Face Verification. Using OpenCV we took the 30 seconds video of person extracting 100 images of their face and using that dataset to test the face on pancard. 
