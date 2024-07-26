# Mean-Average-Precision-for-Object-Detection-COCO-format-Json-
Calculate mean average precision of object detection model manually. Instead of using builtin function of COCO you can use my code for calculating the mAP of an Object detection model. This code calculate mAP using Json files in COCO format

<br>
<h2>This code takes two json files in COCO format as input</h2>
<ul>
  <li>Ground_truth Json file containing bounded boxes</li>
  <li>predictions Json file containing bounded boxes</li>
</ul>

<h3>I have uploaded the sample ground_truth file and prediction file</h3>

<h4>Thresholds For mAP</h4>
<p>This code computes the mAP at single threshold. You can change the value of threshold to compute mAP at multiple thresholds. Or you can modify the logic by providing a list of thresholds and looping all code for each threshold and at end you can take aveerage of all APs</p>
