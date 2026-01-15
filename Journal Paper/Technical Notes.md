# SALAMANDER
## Motivation
Systems of interest have partially known dynamics. Unknown forces can lead to simulation biases, which basically means our model results are skewed. As we move toward relying more on autonomous systems in complex missions, it is important to be able to model these unknown forces and correct the base level of dynamics. Neural networks have been used to perform orbit prediction, but they are completely data driven which is data inefficient and the models are longer to train. 

Deep symbolic regression has been used to uncover unmodeled forces, but the authors argue that Universal Differential Equations (UDEs) are more suitable because they are more expressive, robust, and accurate than symbolic regression. To this end, they combine symbolic regression and UDEs. 

Takeaways:
- Good motivation here for why we want models to predict unknown dynamics *on top* of the base dynamics - this is more data efficient and utilizes laws that we already know. 
- What are UDEs? 

## Background
### SciML
Mechanistic models are derived from known physical laws. You probably know the structure of the system and use dasta to estimate parameters. Non-mechanistic models are data driven and are black box predictive models which do not need knowledge of the system. However, they lack interpretability and are not strongly generalizable. Author proposes combining these
