# Forward Models in the Cerebellum using Reservoirs and Perturbation Learning

Code for training a forward model as described in Schmid, K., Vitay, J. and Hamker, F. (2019). "Forward Models in the Cerebellum using Reservoirs and Perturbation Learning". *CCN 2019*.

The reservoir is based on the model in Roessert, C., Dean P. and Porrill, J. (2015). “At the edge of chaos: how cerebellar granular layer network dynamics can provide the basis for temporal filters.” *PLoS computational biology*, 11(10). [doi.org/10.1371/journal.pcbi.1004515](https://doi.org/10.1371/journal.pcbi.1004515).

Perturbation learning is based on the algorithm described in Bouvier, G. et al. (2018). “Cerebellar learning using perturbations.” *eLife* 7. [doi.org/10.7554/eLife.31599](https://doi.org/10.7554/eLife.31599).


## Dependencies

* [ANNarchy](https://bitbucket.org/annarchy/annarchy) - a neural simulator
* [numpy](https://numpy.org/) - package for scientific computing with Python
* [matplotlib](https://matplotlib.org/) - 2D plotting library


## Run Simulations

To train and test the network on the provided data, execute the following command:

```
python start.py training_and_test_sets.npz <output directory>
```
Results will be saved in npz-format to &lt;output directory&gt;.


## Plot Results

To plot the results, run:

```
python plot_mean_error.py <results.npz>
python plot_responses.py <results.npz>
python plot_trajectory.py <results.npz>
python plot_pn_activity.py <results.npz>

```







