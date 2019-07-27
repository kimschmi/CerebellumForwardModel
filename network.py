from ANNarchy import *
import numpy as np
import math

number_in = 4
number_gc = 1000
number_goc = 100
number_pn = 2
number_pc = number_pn * 10
number_io = number_pn
sparsity_factor = 1
number_gc_pc = int(number_gc/number_pc*number_pn/sparsity_factor)
tau_gc_goc = 1
tau_goc_gc = 50
convergence_gc_goc = 10
convergence_goc_gc = 4
mean_gc_goc = 50 / tau_gc_goc
var_gc_goc = 4 * mean_gc_goc
mean_goc_gc = 0.05 / tau_goc_gc
var_goc_gc = 0 * mean_goc_gc
d_stim = 20
d_delay = 50
d_response = 5
number_training_examples = 5000
number_epochs = 2000
number_test_examples = 5000


neuron = Neuron(
    parameters="""
        g = 1.0 : population
        tau = 1.    : population    # Time constant
        f = 1.0    # Push-pull factor
     """,
    equations="""
        # Integrate firing rate over time
        r = rate + exp(-1/tau) * r
        # Firing rate
        rate = g * sum(in) + sum(inh) + sum(exc) : min=0.0
    """
)

purkinje_cell = Neuron(
    parameters="""
        error_change = 0.0
        tau = 1. : population    # Time constant
    """,
    equations="""
        # Integrate firing rate over time
        r = rate + exp(-1/tau) * r
        perturbation = sum(perturb)
        # Firing rate
        rate = sum(exc) + perturbation : min=0.0
    """
)

inferior_olive_neuron = Neuron(
    parameters="""
        tau = 1. : population    # Time constant
        frequency=50.0 : population    # Frequency of perturbation
        amplitude=0.1 : population    # Amplitude of perturbation
        start_perturb=0.0 : population    # Control when to apply perturbations
    """,
    equations="""
        # Integrate firing rate over time
        r = rate + exp(-1/tau) * r
        # Random perturbations
        rate = if Uniform(0.0, 1.0) < frequency/1000 : 
                            start_perturb * amplitude
                        else : 
                            0.0
    """
)

mossy_fibre = Synapse(
    parameters="""
        baseline = 1.0
    """,
    equations="""
        potential = baseline + baseline * post.f * pre.r : min = 0.0
    """,
    psp="""
        potential 
    """
)

gc_pc_synapse = Synapse(
    parameters="""
        learning_phase = 0.0 : projection   # Update weights at end of trial
        eta = 2e-4  : projection         # Learning rate
        max_weight_change = 5e-1: projection     # Clip weight changes
        start_trace = 0.0 : projection      # Control when to compute eligibility trace
    """,
    equations="""
        # Eligibility trace
        trace += if learning_phase < 0.5 : 
                        start_trace * pre.r * post.perturbation
                    else : 
                        0.0

        # Weight update
        delta_w = if learning_phase > 0.5 : 
                    eta * trace * post.error_change         
                else : 
                    0.0      : min = -max_weight_change, max = max_weight_change
        w -= delta_w : min = 0.0   
    """
)

gc_me_synapse = Synapse(
    parameters="""
        learning_phase = 0.0 : projection   # Update weights at end of trial
        eta = 0.0002  : projection         # Learning rate
        max_weight_change = 5e-1: projection     # Clip weight changes
        start_trace = 0.0 : projection      # Control when to compute eligibility trace
    """,
    equations="""
        # Eligibility trace
        trace += if learning_phase < 0.5 : 
                    start_trace * pre.r
                else : 
                    0.0

        # Weight update
        delta_w = if learning_phase > 0.5 : 
                eta * trace * post.error_change         
            else : 
                0.0      : min = -max_weight_change, max = max_weight_change
        w += delta_w    
  """
)

input = Population(number_in, Neuron("r=0.0"), name="input")
granule_cells = Population(number_gc, neuron, name="granule_cells")
golgi_cells = Population(number_goc, neuron, name="golgi_cells")
purkinje_cells = Population(number_pc, purkinje_cell, name="purkinje_cells")
projection_neurons = Population(number_pn, neuron, name="projection_neurons")
inferior_olive_neurons = Population(number_io, inferior_olive_neuron, name="inferior_olive_neurons")
mean_error_neuron = Population(1, purkinje_cell, name="mean_error_neuron")

inp_gc = Projection(input, granule_cells, synapse=mossy_fibre, target="in", name="inp_gc")
inp_gc.connect_fixed_number_pre(1, weights=1.0)
inp_goc = Projection(input, golgi_cells, synapse=mossy_fibre, target="in", name="inp_goc")
inp_goc.connect_fixed_number_pre(1, weights=1.0)
inp_pn = Projection(input, projection_neurons, synapse=mossy_fibre, target="in", name="inp_pn")
inp_pn.connect_fixed_number_pre(1, weights=1.0)

gc_goc = Projection(granule_cells, golgi_cells, target="exc", name="gc_goc")
gc_goc.connect_fixed_number_pre(convergence_gc_goc, weights=Normal(mean_gc_goc, var_gc_goc),
                                force_multiple_weights=True)

goc_gc = Projection(golgi_cells, granule_cells, target="inh", name="goc_gc")
goc_gc.connect_fixed_number_pre(convergence_goc_gc, weights=Normal(mean_goc_gc, var_goc_gc),
                                force_multiple_weights=True)


gc_pc_connect = np.empty((number_pc, number_gc), object)
for p in range(number_pc):
    index_factor = p % int(number_pc/number_pn)
    gc_connect = np.empty(number_gc_pc * sparsity_factor, object)
    gc_connect[: number_gc_pc] = np.random.normal(0.1, 0.05, number_gc_pc)
    np.random.shuffle(gc_connect)
    gc_pc_connect[p, index_factor * number_gc_pc * sparsity_factor: (index_factor+1) * number_gc_pc * sparsity_factor] = gc_connect
gc_pc = Projection(granule_cells, purkinje_cells, target="exc", synapse=gc_pc_synapse, name="gc_pc")
gc_pc.connect_from_matrix(gc_pc_connect)

io_pc_connect = np.empty((number_pc, number_io), object)
io_pc_connect[:10, 0] = 1.0
io_pc_connect[10:, 1] = 1.0
io_pc = Projection(inferior_olive_neurons, purkinje_cells, target="perturb", name="io_pc")
io_pc.connect_from_matrix(io_pc_connect)

pc_pn_connect = np.empty((number_pn, number_pc), object)
pc_pn_connect[0, :10] = -0.1
pc_pn_connect[1, 10:] = -0.1
pc_pn = Projection(purkinje_cells, projection_neurons, target="inh", name="pc_pn")
pc_pn.connect_from_matrix(pc_pn_connect)

gc_me = Projection(granule_cells, mean_error_neuron, target="exc", synapse=gc_me_synapse, name="gc_me")
gc_me.connect_fixed_probability(0.5, weights=Uniform(0.0, 0.1), force_multiple_weights=True)

monitor_pn = Monitor(projection_neurons, 'rate')
monitor_gc = Monitor(granule_cells, 'rate')
monitor_me = Monitor(mean_error_neuron, 'rate')


def create_network(pn_baselines):
    """Create cerebellar network.

    The network is based on the model described in Roessert, C., Dean, P. and Porrill, J. (2015). "At the Edge of Chaos:
    How Cerebellar Granular Layer Network Dynamics can Provide the Basis for Temporal Filters." PLoS Computational
    Biology, 11(10):e1004515.

    :param pn_baselines: baseline MF input to PN
    :return: network instance
    """
    net = Network(everything=True)
    net.compile()
    inp_gc = net.get_projection("inp_gc")
    inp_goc = net.get_projection("inp_goc")
    inp_pn = net.get_projection("inp_pn")
    gc_goc = net.get_projection("gc_goc")
    goc_gc = net.get_projection("goc_gc")
    gc_pc = net.get_projection("gc_pc")
    granule_cells = net.get_population("granule_cells")
    golgi_cells = net.get_population("golgi_cells")

    for dendrite in inp_gc.dendrites:
        dendrite.baseline = np.random.normal(1.2, 0.1)

    for dendrite in inp_goc.dendrites:
        dendrite.baseline = np.random.normal(1.2, 0.1)

    for i in range(number_pn):
        dendrite = inp_pn.dendrite(i)
        dendrite.baseline = pn_baselines[i]
        print(dendrite.baseline)

    for dendrite in gc_goc.dendrites:
        dendrite.w = [0.0 if w < 0.0 else (w * 2 / convergence_gc_goc) for w in dendrite.w]

    for dendrite in goc_gc.dendrites:
        dendrite.w = [0.0 if w < 0.0 else (w * 2 / convergence_goc_gc) for w in dendrite.w]
        dendrite.w = [-1. * w for w in dendrite.w]

    for dendrite in gc_pc.dendrites:
        dendrite.w = [0.0 if w < 0.0 else w for w in dendrite.w]

    golgi_cells.tau = tau_goc_gc
    granule_cells.tau = tau_gc_goc
    golgi_cells.f = np.random.choice([-1.0, 1.0], number_goc)
    granule_cells.f = np.random.choice([-1.0, 1.0], number_gc)

    return net


def reset(net):
    """Reset network between trials.

    :param net: network to be reset
    :return:
    """
    granule_cells = net.get_population("granule_cells")
    golgi_cells = net.get_population("golgi_cells")
    input = net.get_population("input")
    inferior_olive_neurons = net.get_population("inferior_olive_neurons")
    gc_me = net.get_projection("gc_me")
    gc_pc = net.get_projection("gc_pc")
    granule_cells.rate = np.random.uniform(0.0, 0.1, number_gc)
    granule_cells.r = np.random.uniform(0.0, 0.1, number_gc)
    golgi_cells.rate = np.random.uniform(0.0, 0.1, number_goc)
    golgi_cells.r = np.random.uniform(0.0, 0.1, number_goc)
    input.r = .0
    inferior_olive_neurons.start_perturb = .0
    gc_pc.start_trace = .0
    gc_me.start_trace =.0
    gc_pc.trace = .0
    gc_me.trace = .0


def train(net, inputs, targets, pn_baselines, arm_length):
    """Train cerebellar network using perturbation learning.

    Perturbation learning is based on the algorithm described in Bouvier, G. et al. (2018). "Cerebellar Learning using
    Perturbations". eLife 7:e31599.

    :param net: network to be trained
    :param inputs: samples consisting of x_prev, y_prev, delta theta_0, delta theta_1
    :param targets: x_next, y_next
    :param pn_baselines: baseline MF input to PN, required for mapping network response to target range
    :param arm_length: length of arm segments l_0, l_1
    :return: recordings of error and error estimate, PN response to selected input samples mapped to target range
    """
    input = net.get_population("input")
    purkinje_cells = net.get_population("purkinje_cells")
    mean_error_neuron = net.get_population("mean_error_neuron")
    inferior_olive_neurons = net.get_population("inferior_olive_neurons")
    gc_pc = net.get_projection("gc_pc")
    gc_me = net.get_projection("gc_me")
    mon_pn = net.get(monitor_pn)
    mon_gc = net.get(monitor_gc)
    mon_me = net.get(monitor_me)
    mon_gc.pause()
    mon_pn.start()
    mon_me.start()

    number_training_examples = np.size(inputs, axis=0)
    error_estimates = np.zeros(int(number_epochs * number_training_examples / 2))
    training_errors = np.zeros(int(number_epochs * number_training_examples / 2))
    current_errors = np.zeros(number_training_examples)
    error_index = 0
    responses = np.zeros((int(number_training_examples / 10), number_epochs, number_pn))

    try:
        for trial in range(number_epochs * number_training_examples):
            reset(net)

            example = trial % number_training_examples

            input.r = inputs[example, :]
            net.simulate(d_stim)
            input.r = 0.0
            net.simulate(d_delay - 1)

            # Begin computing eligibility trace
            gc_pc.start_trace = 1.0
            gc_me.start_trace = 1.0
            net.step()

            # Begin applying perturbations
            if trial < (number_epochs - 2) * number_training_examples:
                inferior_olive_neurons.start_perturb = 1.0

            net.simulate(d_response)

            # Sample PN response
            rates_pn = mon_pn.get('rate')
            response = rates_pn[-d_response:, :]
            response = np.mean(response, axis=0)
            # Map PN response to target range
            mapped_response = (response - (pn_baselines / 4)) / (pn_baselines / 2)
            mapped_response = mapped_response * 2 * np.sum(arm_length) - np.sum(arm_length)

            # Record PN responses
            if example < number_training_examples / 10:
                responses[example, int(trial / number_training_examples), :] = mapped_response

            # Sample error estimate
            rates_me = mon_me.get('rate')
            error_estimate = rates_me[-d_response:, :]
            error_estimate = np.mean(error_estimate)

            inferior_olive_neurons.start_perturb = 0.0

            error = np.linalg.norm(targets[example, :] - mapped_response)

            # Record training error and error estimate
            if trial % (number_training_examples * 2) < number_training_examples:
                training_errors[error_index] = error
                error_estimates[error_index] = error_estimate
                error_index += 1

            # Update weights
            error_change = math.copysign(1.0, error - error_estimate)
            gc_pc.learning_phase = 1.0
            gc_me.learning_phase = 1.0
            purkinje_cells.error_change = error_change
            mean_error_neuron.error_change = error_change

            net.step()

            # Learning finished
            gc_pc.learning_phase = 0.0
            gc_me.learning_phase = 0.0

            # Clear recordings of last step
            _ = mon_pn.get()

            if trial % number_training_examples == 0:
                print("Mean square error over training set:", np.mean(current_errors))

            if trial % 1000 == 0:
                print("Trial", trial, ": error =", error, "\tmean error =", error_estimate, "\ttarget =", targets[example])

            current_errors[example] = error

    except KeyboardInterrupt:
        # Clear recordings
        _ = mon_pn.get()
        pass

    return training_errors, error_estimates, responses


def test(net, inputs, targets, pn_baselines, arm_length):
    """Test network performance.

    :param net: network to be tested.
    :param inputs: input samples consisting of x_prev, y_prev, delta theta_0, delta theta_1
    :param targets: x_next, y_next
    :param pn_baselines: baseline MF input to PN, required for mapping PN response to target range
    :param arm_length: length of arm segments l_0, l_1
    :return: recordings of error and PN response mapped to target range, PN activity during trial for selected samples
    """
    input = net.get_population("input")
    mon_pn = net.get(monitor_pn)
    mon_gc = net.get(monitor_gc)
    mon_me = net.get(monitor_me)
    mon_gc.pause()
    mon_me.pause()
    mon_pn.start()

    number_test_examples = np.size(inputs, axis=0)

    mapped_responses = np.zeros((number_test_examples, number_pn))
    errors = np.zeros(number_test_examples)
    pn_activity = np.zeros((int(number_test_examples / 100), d_stim + d_delay + d_response, number_pn))
    responses = np.zeros((number_test_examples, number_pn))

    for test in range(number_test_examples):
        reset(net)

        input.r = inputs[test, :]
        net.simulate(d_stim)
        input.r = 0.0
        net.simulate(d_delay)
        net.simulate(d_response)

        # Sample PN response
        rates_pn = mon_pn.get('rate')
        response = rates_pn[-d_response:, :]
        response = np.mean(response, axis=0)
        # Map PN response to target range
        mapped_response = (response - (pn_baselines / 4)) / (pn_baselines / 2)
        mapped_response = mapped_response * 2 * np.sum(arm_length) - np.sum(arm_length)
        mapped_responses[test, :] = mapped_response
        responses[test, :] = mapped_response

        error = np.linalg.norm(targets[test] - mapped_response)
        errors[test] = error

        if test < number_test_examples / 100:
            pn_activity[test, :] = rates_pn

        if test % 1000 == 0:
            print("Trial", test, ": error =", error)

    return errors, responses, pn_activity


