+++
author = "ipark"
title = "Lighter Faster Semantic Segmentation by Post Training Quantization and Quantization Aware Training"
draft =  false
type = "projects"
layout = "projects"
description = ""
tags = ["project", "Semantic Segmentation", "Deep Learning", "MobileNet",
"DeepLabV3+", "Quantization", "Tensorflow",
"Tensorflow Light", "Computer Vision", "AWS"
]
+++
<h3 id="summary">Summary</h3>
<p>Experimenting with Quantization of Tensorflow Models on various datasets with the DeepLab v3 Decoder architecture and MobileNet v2 Encoder architecture using a variety of techniques including 
<ul>
  <li>Quantization aware training </li>
  <li>Quantization aware training with delay </li>
  <li><a href="https://gitlab.com/ipark/cs256-ai/blob/master/ImageSegmentationWithDeeplab/CS256_GroupE_PostQuantization.ipynb">Post training Quantization</a></li>
  <li> <a href="https://gitlab.com/ipark/cs256-ai/blob/master/ImageSegmentationWithDeeplab/CS256_GroupE_inference_deeplab.ipynb">Quantized Inference/Evalulation </a></li>
</ul>
</p>
<h3 id="presentation-slide">Presentation Slide</h3>
<ul>
  <li><a href="https://gitlab.com/ipark/cs256-ai/blob/master/ImageSegmentationWithDeeplab/docs/CS256_GroupE_Final_Presentation.pdf">
     Lighter, Faster Semantic Segmentation by Post-Training Quantization and Quantization-Aware Training</a>
  </li>
</ul>
<h3 id="deeplab-deep-labelling-for-semantic-image-segmentation">DeepLab: Deep Labelling for Semantic Image Segmentation</h3>
<p>@inproceedings{deeplabv3plus2018,
title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
booktitle={ECCV},
year={2018}
}</p>
<h3 id="installation">Installation</h3>
<p>pip install all the following required packages.</p>
<h3 id="requirement">Requirement</h3>
<ul>
  <li>TensorFlow 1.15</li>
  <li>Jupyter Notebook</li>
  <li>Python 3.6</li>
  <li>Numpy</li>
  <li>Pillow</li>
  <li>matplotlib</li>
  <li>conda</li>
</ul>
<p>Note: For a ready to use envirenment, a deeplearning ami on an EC2 instance would come with all the required packages needed to run this repo immediatly. </p>
<h3 id="usage-on-colab">Usage on Colab</h3>
<ul>
 <li>Fine-tuning and Quantization</li>
 <img src="https://raw.github.com/SherifSabri/ImageSegmentationWithDeeplab/master/quantize.png" width="80%">
 <li>Inference</li>
 <img src="https://raw.github.com/SherifSabri/ImageSegmentationWithDeeplab/master/inference.png" width="80%">
</ul>
<h3 id="usage-on-aws">Usage on AWS</h3>
<ul>
  <li>clone the repo</li>
  <li>navigate to ImageSegmentationWithDeeplab (command: cd ImageSegmentationWithDeeplab)</li>  
  <li>run the command "jupyter notebook"</li>  
  <li>use the provided url (default: localhost:8888)</li>
  <li>open the "inference_deeplab.ipynb" notebook</li>
  <li>From drop down list Cell > Run All </li>
</ul>
<h3 id="results">Results</h3>
<ul>
 <li>FLOAT32 Segmentation</li>
 <img src="https://raw.github.com/SherifSabri/ImageSegmentationWithDeeplab/master/mobileNetv2-f32.png" width="80%">
 <li>Post-Quantization UINT8 Segmentation (no fine-tuning) </li>
 <img src="https://raw.github.com/SherifSabri/ImageSegmentationWithDeeplab/master/postQuant-8bit-noFT.png" width="80%">
 <li>Post-Quantization UINT8  Segmentation (10K-iteration fine-tuning) </li>
 <img src="https://raw.github.com/SherifSabri/ImageSegmentationWithDeeplab/master/postQuant-8bit-10kFT.png" width="80%">
 <li>Quantization-Aware-Training UNIT8 Segmentation </li>
 <img src="https://raw.github.com/SherifSabri/ImageSegmentationWithDeeplab/master/QAT-8bit.png" width="80%">
</ul>
<h3 id="about">About:</h3>
<p>This page (code, report and presentation) is the group "E" submission for Final project for CS256: Selected Topics in Artificial Intelligence, Section 2. Leb by Instructor: Mashhour Solh, Ph.D.
</br>
The group members are:
<ul>
  <li>Sherif Elsaid</li>
  <li>Inhee Park</li>
  <li>Sagar Shahi</li>
  <li>Sriram Priyatham Siram</li>
  <li>Anand Vishwakarma</li>
</ul>
The code maybe used for educational and commercial use under no warranties. 
</br>For questions on this project and code please reach out to: 
</br>"contact@sherifsabri.dev"
<h3 id="credits">Credits</h3>
<p>This project was conducted with free credits provided by AWS educate team.</p>
