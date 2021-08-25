
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Multiple Dataset Methods Classifier</h3>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#Limitations">Limitations</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Wonder what scores of precision can you achieve with different angles of data classification?
So in this project I managed to create semi-automat classifier of dataset (Numeric/Categorial) data.

This classifier includes:
* Naive Bayes
* Decision Tree
* K-Nearest Neighbors
* K-Means

For Naive Bayes and Decision Tree there are 2 options, one is written by me, The other uses an api of SKLearn library.
K-Nearest Neighbors and K-Means do not use SKLearn.

Evaluator.py File:

You might want to to evaluate the score of the given dataset, So I included an Evaluator.py file which automatically applied those tests for every classification made.
* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-Score
* Test File Accuracy
* Train File Accuracy

Preprocessing.py File:
* This file contains all of the background common and shared functions between the classifiers.
* You might find functions related to Discretization, Normalization, Entropy and Probabilities calculations.

PickleFiles.py File:
* An option to save the current results or load the instances of classes from pickle files is added in order to save some process time.
* Dedicated folder for pickle files are automatically created after a successfully classification.

## Built With
* [Python 3.8.0](https://www.python.org/downloads/release/python-380/)



<!-- GETTING STARTED -->
## Getting Started
In order to use the classifier, Python V3.8.0 or above is needed.

## Prerequisites

The libraries below must be installed in order to use the classifier 
* os-sys
  ```sh
  pip install os-sys
  ```
* pandas
  ```sh
  pip install pandas
  ```
* entropy-based-binning
  ```sh
  pip install entropy-based-binning
  ```
* numpy
  ```sh
  pip install numpy
  ```
* pickle5
  ```sh
  pip install pickle5
  ```
* collection
  ```sh
  pip install collection
  ```
* sklearn
  ```sh
  pip install -U scikit-learn
  ```
* math
  ```sh
  pip install math
  ```
* seaborn
  ```sh
  pip install seaborn
  ```
* matplotlib
  ```sh
  pip install matplotlib
  ```

<!-- USAGE EXAMPLES -->
## Usage
Download python file to any location desired. 
Go to Driver.py file and start [run()] fucntion, Here is an example:
<br></br>
<a href="https://ibb.co/HBv8FQs"><img src="https://i.ibb.co/YP9sRGM/image.png" alt="image" border="0"></a>

<!-- Limitations -->
## Limitations
* Train file and Test file must be .csv files.
* Dataset file is not splited automatically into 70% train, 30% test data, files must be separated prior the run.
* Classifier support only 1 classification column.
* Classification algorithms are not fully optimized. Therefore, the bigger the files are, the bigger the classification time will take.
* During the run, if inputs are manipulated there might be a chance of failure or undesired result. Error mechanism was built for 95% of the inputs.



<!-- CONTRIBUTING -->
## Contributing
Mant thanks to my classmate Tamar - [@Tamar](https://github.com/tamar1472) for making this project possible.

<!-- LICENSE -->
## License

`unlicense` at the moment.



<!-- CONTACT -->
## Contact

Liran - [@Liran](https://www.linkedin.com/in/liran-smadja/)

Other Projects Link: [Projects](https://github.com/liran121211)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub README Template](https://github.com/othneildrew/Best-README-Template)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/liran121211/Dataset_Classifier/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/liran-smadja/
[product-screenshot]: images/screenshot.png
