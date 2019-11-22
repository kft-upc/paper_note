# [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115)

------

## QUESTION  
* How to set the weights for each of task's loss automatically.
* Do the Semantic, instance, depth segmentation by the multi-task leaning networks.
## Method
- Prior approaches.

    $$
    L_{total} = \sum_{i=1}^{K} \omega _i L_i
    $$

    $$ L_{total} $$ is the loss function of all the tasks. $i$ represent the different task. This is just a linear sum of the losses of all the task. The weight $\omega _i$ is set as uniform or just manually tuned. This way is not effective and very expensive.  
  
- Present approach.
	1. Based on the Bayesian model, the author present three kind of uncertainty.
	* Epistemic uncertainty: reflect the model can not learn due to the lack of train data. It's with respect to the data.
	* Aleatoric uncertainty: reflect the model can not learn due to our data can not explain. So it's with respect to the model.
	    * Data-dependent: with respect to the input data.
	    * Task-dependent (**Homoscedastic**): with respect to the task and remain constant with the different input data. 
	* **Homoscedastic uncertainty** can be used for the calculate the weight.
	2. Give the multi-likelihood
	* Regression tasks (use the Gaussian distribution)
		$$
		p(y|f^{W}(x)) = N(f^W(x), \sigma) 
		$$
		The output of the network $f^W(x)$ of the input $x$ and weights $W$ is set as the mean of the Gaussian distribution. The $\sigma$ is the observation noise scalar, which is the learnable parameter when train the network.
		For the multi-task model. 
		$$
		p(y_1,...,y_K|f^W(x)) = p(y_1|f^W(x))\cdot \cdot \cdot p(y_K|f^W(x)) 
		$$
		The likelihood can be expressed as the multiple of all the task's likelihood.
		For the model learning, we need to max the log likelihood of the network. The log likelihood for a task can be expressed as below:
		$$
		\log p(y|f^W(x)) \propto -\frac{1}{2\sigma^2}\left \|y-f^W(x)\right \|^2 - \log \sigma
		$$
		
		Then we need to maximise the log likelihood with respect to the model parameters $W$ and noise scale $\sigma$.
		In the training of the model, we need to minimize the loss function $L(W, \sigma_1, \sigma_2)$.
		$$
		L(W, \sigma_1, \sigma_2) = -\log p(y_1,y_2|f^W(x)) \\
		\propto \frac{1}{2\sigma_1^2}\left \| y_1 - f^W(x) \right \|^2 + 
		\frac{1}{2\sigma_1^2}\left \| y_1 - f^W(x) \right \|^2 + \log\sigma_1\sigma_2 \\
		= \frac{1}{2\sigma_1^2}L_1(W)+\frac{1}{2\sigma_1^2}L_2(W)+\log\sigma_1\sigma_2
		$$
		the $L_1(W)$ is the loss for the first task. It's the similar as the second one.
		The learning parameters $\sigma_1$ and $\sigma_2$ learn the noise of different task and then can be the realitive weight of the losses. 
	* Classification problem
	The likelihood with the softmax function.
	$$
	p(y|f^W(x), \sigma) = Softmax(\frac{1}{\sigma^2}f^W(x))
	$$
	The softmax funciton is :
	$$
	softmax(x)=\frac{e^{x_i}}{\sum _{i=1} ^K e^{x_i}}
	$$
	So the log likelihood is:
	$$
	\log p(y|f^W(x), \sigma) = \frac{1}{\sigma^2}f^W(x) - \\
	\log \sum_
	
		
		
		
		






## Word
- trivially : easily.
- squash: to compress with violence, out of natural shape or condition

​	
​		
​	



