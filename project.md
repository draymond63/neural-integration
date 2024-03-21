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
1. How is the associative memory being used? Can the given landmark map be wrong, and the memory corrects it?
2. How are landmarks being identified? How does the model know that landmark A is not landmark B?
   - The index of the closest landmark is immediately given to the network as `landmark_id_input` (which directly corresponds to the function `landmark_id_func`)
3. Why is velocity the input? Shouldn't it be acceleration?
4. Can the agent rotate? Or only move translationally?
5. Whats the point of cleanup?
   - Smooths PI output using grid cells
6. Inuitive understanding of SSP Space?
7. PI model is kinda like an equivalent to a lone IMU?

### Output
7. In `run_slam`, why is the path integrator so bad? It seems like the localization is entirely due to landmark corrections. It's better in lone PI trial.
8. In `slam_vs_pi_trials`, PI overshoots but also has sharp discontinuities. Why?
9. Both SLAM and PI update pretty infrequently. Why?


# Project Plan
Compare Kalman filter to neural path integration
 - Add uncertainty integration to SSP encoding
 - 


# Ideas
- Generalized phase correlation that can handle rotation of the visual field?


```
cd experiments/
python run_slam.py --domain-dim 2 --seed 0  --save True --plot True --save_plot True --ssp_dim 55 --pi_n_neurons 500
```


Sources
- https://learn.uwaterloo.ca/d2l/le/content/999135/viewContent/5402333/View
- https://chat.openai.com/c/e104cbcc-2e53-4f57-9e93-a9904db83692
- [Visual Odometry](https://en.wikipedia.org/wiki/Visual_odometry)
- https://docs.px4.io/main/en/computer_vision/visual_inertial_odometry.html
- [Kalman Textbook](https://drive.google.com/file/d/0By_SW19c1BfhSVFzNHc0SjduNzg/view?resourcekey=0-41olC9ht9xE3wQe2zHZ45A)