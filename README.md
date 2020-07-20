## Solution for ALASKA2 Image Steganalysis: Detect secret data hidden within digital images (Working in Progress)

This is a solution for competition to detect secret data hidden within innocuous-seeming digital images. Current methods to detect secret data hidden within innocuous-seeming digital images that produce unreliable results, raising false alarms. This kind of solution can helps haw enforcement officers to combat criminals using hidden messages. More accurate methods could help identify communications from criminals whose hide information in plain sight.

The challenge is organized by Rémi COGRANNE (UTT), Patrick BAS (CRIStAL / CNRS) and Quentin Giboulot (UTT). The IEEE WIFS (Workshop on Information Forensics and Security) has teamed up with Troyes University of Technology, CRIStAL Lab, Lille University, and CNRS to enable more accurate steganalysis. WIFS is an annual event where researchers gather to discuss emerging challenges, exchange fresh ideas, and share state-of-the-art results and technical expertise in the areas of information security and forensics. WIFS 

To increase accuracy, the method data hidden within digital images “into the wild” (hence the name ALASKA) to mimic real world conditions.

Competition Site: https://www.kaggle.com/c/alaska2-image-steganalysis


### Evaluation Metrics
The submissions are evaluated on the weighted AUC to focus on reliable detection with an emphasis on low false-alarm rate. To calculate the weighted AUC, each region of the ROC curve is weighted according to these chosen parameters:
```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

### Data Description
 Rather than limiting the data source, these images have been acquired with as many as 50 different cameras (from smartphone to full-format high end) and processed in different fashions. This dataset contains a large number of unaltered images, called the "Cover" image, as well as corresponding examples in which information has been hidden using one of three steganography algorithms (`JMiPOD`, `JUNIWARD`, `UERD`). The goal of the competition is to determine which of the images in the test set (Test/) have hidden messages embedded.
    Cover/ contains 75k unaltered images meant for use in training.
    JMiPOD/ contains 75k examples of the JMiPOD algorithm applied to the cover images.
    JUNIWARD/contains 75k examples of the JUNIWARD algorithm applied to the cover images.
    UERD/ contains 75k examples of the UERD algorithm applied to the cover images.
    Test/ contains 5k test set images. These are the images for which you are predicting.


### Solution

The solution code provide: `trainer` and `stacker` to generate accurate predictions. It leverage the ensemble and stacking classifier using prediction from the modern deep learning model `efficinetnet` and `seresnext`.
`trainer`: training deep learning model with the popular architectures and pretrained weights.
`stacker`: to generate submissions using the prediction from `trainer`.
