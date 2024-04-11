# Summary
1. Build Neural localization model
   - Vestibular System (IMU)
   - Vision + Knowledge of landmark locations
   - Associative Memory
2. Build Kalman-based localization model: Visual inertial odometry
   - IMU (3 axis acceleration + 3 axis rotation rate)
   - Vision (https://en.wikipedia.org/wiki/Visual_odometry)
     - Estimates relative position of known landmarks
3. Compare ability to keep track of position and orientation relative to landmarks 


# Code Breakdown
## Components
- Path Integrator
  - Uses hexagonal grid cells
- Associative Memory
  - Self-position?
  - Inputs: key_input (lm_id), value_input (lm_vec), learning (lm_in_view)
  - Outputs: recall
- Circular Convolution
  - Method of querying landmark locations?

## Logic
- Closest landmark ID (index) is available immediately, along with the correct vector too it
  - ID is given to associative memory's `key_input`
  - Vector to landmark is encoded and given to:
    - landmark_ssp_ens (`CircularConvolution`) with pathintegrator output to estimate global position of landmarks
    - position_estimate (`CircularConvolution`) with associative memory output to estimate self-position
- Position estimate and path integrators outputs (as SSPs) are given to `update_state`
  - If dot product of SSPs is above threshold, send different to path integrator


# Questions
1. How are landmarks being identified? How does the model know that landmark A is not landmark B?
   - The index of the closest landmark is immediately given to the network as `landmark_id_input` (which directly corresponds to the function `landmark_id_func`)
2. Why is velocity the input? Shouldn't it be acceleration?
   - Velocity exists in the brain
3. Can the agent rotate? Or only move translationally?
   - State is x & y, nothing else
4. Whats the point of cleanup?
   - Smooths PI output to snap to grid
5. In `run_slam`, why is the path integrator so bad? It's better in lone PI trial.
6. In `slam_vs_pi_trials`, PI overshoots but also has sharp discontinuities. Why?
   - Decoding maps the same pointer to an infinite number of x, y values.
7.  Both SLAM and PI update pretty infrequently. How would I increase the resolution?
   - Byproduct of decoding. Change `num_samples` in `SSPSpace.decode`


## For Furlong
1. Cosine similarity already has diffusion. Should I use a different distance metric? Or train something that interprets the cosine similarity grid to extract covariance?
   - Set length scale to make the output much more precise
   - Square and normalize distribution to get pdf
   - Calculate covariance matrix of nonnormal pdf

2. Order of operations
   - noise, encode, sum: f(x2) = f(x1) * SUM f(dxi)

3. Should averaging be done using `transform` or `function`? Is there value in the approximation?
4. `nengo.Neuron`: Is the base model simply using a tuning curve to calculate fire rates? What happens when you use other models?
5. Summation post-encoding but pre-log?
6. Code does log of Fourier (`ssp_space, ln 225`), but math should be Fourier of log?

# Project Plan
Compare Kalman filter to neural path integration
 - Add uncertainty integration to SSP encoding
 - \phi(x_{t+1}) = \phi(x_t) \circledast [\phi(x_t) + log(\sum_{j}^{n}\phi(\dot{x}_{t,j}))]
 - \phi(x_{t+1}) = F^{-1}\{F\{\phi(x_t)\} \cdot [F\{\phi(x_t)\} + F\{log(\sum_{j}^{n}\phi(\Delta x_{t,j}))\}]\}

 - \phi(x_{t+1}) = F^{-1}\{\phi_F(x_t) \cdot [\phi_F(x_t) + \Delta x_tA_F]\}
 - A_F = Re\{log(F\{I\})\}


# Ideas
- Generalized phase correlation that can handle rotation of the visual field?



Sources
- https://learn.uwaterloo.ca/d2l/le/content/999135/viewContent/5402333/View
- https://chat.openai.com/c/e104cbcc-2e53-4f57-9e93-a9904db83692
- [Visual Odometry](https://en.wikipedia.org/wiki/Visual_odometry)
- https://docs.px4.io/main/en/computer_vision/visual_inertial_odometry.html
- [Kalman Textbook](https://drive.google.com/file/d/0By_SW19c1BfhSVFzNHc0SjduNzg/view?resourcekey=0-41olC9ht9xE3wQe2zHZ45A)