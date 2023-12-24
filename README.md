# FARODIT: Framework for Analysis and Prediction  on Data in Tables

FARODIT is a software that facilitates the development and integration of artificial intelligence technology components, including cloud solution process automation services aimed at modernization, acceleration and adaptation of AI algorithms in the direction of predictive analytics for the digital industry and beyond.
## Goal
We aimed to create a platform that helps machine learning experts quickly and easily develop and test predictive analytics models based on tabular data. The framework provides easy access to data, a set of tools for data preprocessing, and the options for selecting machine learning algorithms and tuning model parameters to maximize predictive accuracy. In addition, the framework is extensible and open to integration with other tools and libraries to meet the needs of different users.
## Functionality
### Data preparation
* tools for cleaning and preprocessing pandas dataframe data
### Model selection
* selection of ML algorithms that can be used to build predictive analytics models
* polynomial neural networks built on the basis of the tensorflow library.
### Training
* model training on the basis of processed data
* parameters tuning parameters and model performance optimization
### Evaluation
* tools to evaluate the accuracy of the model on test data
## Technical specifications
<img src="https://img.shields.io/badge/python-3.9-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/>  <img src="https://img.shields.io/badge/TensorFlow-2.11-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
- handles data of numeric types
- is able to process data with different discreteness 
- can work with a large number of features and measurements
- supports a wide range of machine learning algorithms, including linear regression, decision trees, random forests, gradient bousting, and neural networks, including polynomial networks
## Project Structure
The repository includes the following directories:
* `farodit` contains the main classes and scripts
* `examples` includes several *how-to-use-cases* where you can start to discover how framework works:
	* `airfoil` - simple case of use
	* `industrial_sensors` - case with WindowSlider
## Installation
```
git clone https://github.com/SPbU-AI-Center/farodit.git
cd farodit
pip install -r requirements.txt
```
## Future Development
Any contribution is welcome. Our R&D team is open for cooperation with other scientific teams as well as with industrial partners.
## Contributors
Ovanes Petrosian<br>
Anna Golovkina<br>
Darya Pashkova
