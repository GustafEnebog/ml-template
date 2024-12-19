# Template for ML-repo
Template for a Machine Learning Project Repository

<br>

Welcome,

This is my template that I use for my Machine Learning projects and it's perfectly okay for you to use this template as the basis for your project too.

I have created a starting structure for the readme, prewrtitten Jupyter notebook cells, presinstalled tools, etc.
I have also, before the start of the real part of the README, created some reources for your README and project in general.


## How to use this repo

1. Use this template to create your GitHub project repo

1. Log into your cloud IDE with your GitHub account.

1. On your Dashboard, click on the New Workspace button

1. Paste in the URL you copied from GitHub earlier

1. Click Create

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and `pip3 install -r requirements.txt`

1. Open the jupyter_notebooks directory, and click on the notebook you want to open.

1. Click the kernel button and choose Python Environments.

## Instructions on versions of dependencies

The listing in the requirements-file are given without version number in order to start with the latest version of all dependencies (Python, Jupyter Notebook, streamlit, Pandas, Scikit-learn etc.). 

If you want to "freeze" the version though out the project (often a good idea to prevent changes in new versions to mess up your code) you need to...XXXXXXXXXXxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.
Remember also to always update the python version to the latest version in both the gitpoddockerfile and the python-version!

## On the topic of READMEs

### Other README templates
[awesome-readme](https://github.com/pottekkat/awesome-readme/blob/master/README-template.md)

### Informative Articles on how to write a good README
[How to write an Awesome README](https://towardsdatascience.com/how-to-write-an-awesome-readme-68bf4be91f8b)
[Articles](https://github.com/trekhleb/homemade-machine-learning)
[](https://github.com/pottekkat/awesome-readme/tree/master)
[]()
[]()
[]()
[]()

### Examples of good READMEs
[homemade-machine-learning](https://github.com/trekhleb/homemade-machine-learning)
[]()
[]()
[]()
[]()
[]()

### Principles of a good README (in no particular order)

ASCII and Unicode, capital letters come before lower-case, meaning that README will most often appear at the top of lists of files

start your README file with the most useful information and then gradually work your way down to the technical details.

Who the documentation is for ?

The documentation are intended for you, your coworkers and  the other  users who might be using your code.

Why Documentation ?

It helps collaborating efficient and easy. Plus after some time down the road, you yourselves would have forgotten the line of thinking, for why you did code the way you did. Documentation helps you remember  your line of thinking, not only to yourselves but others too.

Why README ?

It helps anyone who first visits your project to get up to the speed, up and running.

If people don’t know what your software does, then they won’t use it or contribute to it

### Good word to use
Succinct
Verbose

### Markdown (syntax that covers your basic README-needs)

#### Headings
# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6

#### Separators  
You may use *, - or _; it make no difference.

---

#### Empty row
<br>

#### Bulleted items  
You may use *, - or +; it make no difference.
- First level item
  - Second level item
    - Third level item

#### Bold and italics
You may use **, __; it make no difference.  
**Bold**  
You may use *, _; it make no difference.  
*italics*  
You may use ** and * or_; it make no difference.  
***Bold and Italic***  

#### Strikethrough
~~text~~

#### Colored text
<span style="color: red;">This is red text</span>

#### Superscript
Normal text<sup>sup text</sup>

#### Subscript
Normal text<sub>sub text</sub>

#### Escape characters
Syntax: \
Example: To escape a special character like *, use \* → \*escaped text\* will show as *escaped text*.

#### Links
[Google](https://www.google.com)
[]()

#### Background color

#### Tables
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Row 1    | Data     | More data|
| Row 2    | Data     | More data|
| Row 3    | Data     | More data|

[]()

#### Images
Use jpg for: photographs and complex images with gradients and many colors.
Use png for: simple images (e.g. logos/icons), graphics and images with transparency.
![Alt text](https://url-to-image.com/image.png)
link to images:
[![Image](https://url-to-image.com/image.png)](https://link-to-page.com)
[]()

#### Hide stuff
<details><summary>Click to expand</summary>Content here</details>

#### Code
inline code: `this is code`
Code blocks: <pre> ```python def hello(): print("Hello World") ``` </pre>
[]()

#### Blockquotes
> This is a blockquote.

#### Emojis
:sparkles: :rocket:
[]()

#### Equations
LaTeX (MathJax)
Use lightweight KaTeX if performance and fast rendering is an issue 
$E = mc^2$

##### Inline
This is an inline equation: $E = mc^2$.

#### Block Math (Centered Equation)
$$
E = mc^2
$$

#### Fractions
$$
\frac{a}{b} = c
$$

#### Exponents
$$
x^2 + y^2 = z^2
$$

#### Subscripts
$$
a_1 + a_2 = b_1
$$

#### Squareroot
$$
\sqrt{x^2 + y^2}
$$

#### Summation
$$
\sum_{i=1}^{n} i
$$

#### Integrals
$$
\int_0^\infty e^{-x^2} \, dx
$$

#### Matrices
$$
\begin{bmatrix} 
1 & 2 \\
3 & 4
\end{bmatrix}
$$

#### Greek letters
$$
\alpha + \beta = \gamma
$$

#### Aligning Equations
$$
x^2 + y^2 = z^2 \\
a^2 + b^2 = c^2
$$

#### Operators and Functions
$$
\sin(\theta) + \cos(\theta) = 1
$$

### Machine Learning Resources
[Machine Learning Resources](https://github.com/datascienceid/machine-learning-resources/blob/master/README.md)

<br>
<br>
<br>
<br>
<br>
<br>

# THE REAL README TEMPLATE STARTS BELOW. You can safely delete this text and everything above for your readme (when you have read it)!








<div style="text-align: right;">
<img src="images_readme/data_driven_design_logo.png" alt="Logo for Data Driven Design " width=400/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</div>

# Name of Project/Repo
Introductory text


## Table of Contents

- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Hypothesis and how to validate?](#hypothesis-and-how-to-validate?)
- [The rationale to map the business requirements to the Data Visualizations and ML tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Dashboard Design](#dashboard-design)
- [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
- [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
- [Credits](#credits)
- [Acknowledgements](#acknowledgements)


Introduction/Overview
 Project Background
 Problem Statement
 Project Objectives
Dataset Overview
 Dataset Content
 Data Preprocessing
Business Requirements
Hypothesis and Validation
Rationale for Mapping Business Requirements to Visualizations and ML Tasks
ML Business Case
Modeling Approach
 Model Selection
 Hyperparameter Tuning
 Model Evaluation
Results and Evaluation
 Model Performance Metrics
 Comparison of Models
Dashboard Design
Unfixed Bugs
Deployment
Main Data Analysis and Machine Learning Libraries
Ethical Considerations
 Ethical Impact
 Bias and Fairness
 Data Privacy and Security
Scalability and Future Work
 Future Work
 Model Improvement
 Scalability Considerations
Installation/Setup Instructions
 Requirements
 Setup Instructions
 Running the Project
Credits
Acknowledgements
References


## Dataset Content - OPTIONAL
* Describe your dataset. Choose a dataset of reasonable size to avoid exceeding the repository's maximum size and to have a shorter model training time. If you are doing an image recognition project, we suggest you consider using an image shape that is 100px × 100px or 50px × 50px, to ensure the model meets the performance requirement but is smaller than 100Mb for a smoother push to GitHub. A reasonably sized image set is ~5000 images, but you can choose ~10000 lines for numeric or textual data. 


## Business Requirements - OBLIGATORY
* Describe your business requirements


## Hypothesis and how to validate?
* List here your project hypothesis(es) and how you envision validating it (them) 


## The rationale to map the business requirements to the Data Visualizations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualizations and ML tasks


## ML Business Case
* In the previous bullet, you potentially visualized an ML task to answer a business requirement. You should frame the business case using the method we covered in the course 


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements
* Thank the people who provided support through this project.