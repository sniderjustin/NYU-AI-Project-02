

**Monday, March 23rd, 2020  // ? AM** 

* Parameters: 
  * EWC
  * fisher_max= 0.001
  * ewc_lambda = 0.8
    * decreased value 
  * dataset (unrehearsed) clipped= 2000 observations per batch
* Performance: 
  * ?
* Notes: 
  * ?
* Log File:
  * ?



**Monday, March 23rd, 2020  // 9:14 AM** 

* Parameters: 
  * EWC
  * fisher_max= 0.001
  * ewc_lambda = 0.2
    * decreased value 
  * dataset (unrehearsed) clipped= 2000 observations per batch
* Performance: 
  * Task 0 is slightly up, but Task 1 is way down. 
* Log File:
  * 9:14

**Monday, March 23rd, 2020  // 8:50 AM** 

* Parameters: 
  * EWC
  * fisher_max= 0.001
  * ewc_lambda = 0.4
  * dataset (unrehearsed) clipped= 1000 observations per batch
* Performance: 
  * Learning increased from previous settings
* Notes: 
  * Added clipping to fisher values so we can set max fisher value. 
    * hoping this will reduce the dominance of the first round learning. 
  * Changed fisher value update to be half old and half new divided by i + 1 . 
    * I am hoping this will allow new learning to happen faster and with more permanence.
* Log File:
  * 20200323_850AM_EWC.txt

**Monday, March 23rd, 2020  // 8:16 AM** 

* Parameters: 
  * EWC
  * fisher_max= infinity
  * ewc_lambda = 0.4
  * full dataset (unrehearsed)
* Performance: 
  * Bad
* Notes: 
  * Learning rate is too suppressed. 
* Log File: 
  * 20200323_EWC.txt