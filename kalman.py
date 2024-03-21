import numpy as np
import matplotlib.pyplot as plt
from random import *


def ode(state: np.ndarray, inp: np.ndarray):
    # position rate is equal to velocity
    pos_x, pos_y, theta, vel_x, vel_y = state
    dvel, dtheta = inp
    # velocity rate is equal to the accel broken into the xy basis
    dvelx = dvel * np.cos(state[2])
    dvely = dvel * np.sin(state[2])
    return np.array([vel_x, vel_y, dtheta, dvelx, dvely])


def rk4(state, imu, ode, dt):
    # runs a rk4 numerical integration
    k1 = dt * ode(state, imu)
    k2 = dt * ode(state + .5*k1, imu)
    k3 = dt * ode(state + .5*k2, imu)
    k4 = dt * ode(state + k3, imu)

    return state + (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)

def numericalJacobianOfStatePropagationInterface(state, imu, dt):
    # data contains both the imu and dt, it needs to be broken up for the rk4
    return rk4(state, imu, ode, dt)


def numericalDifference(x, data, dt, ep = .001):
    # calculates the numerical jacobian
    y = numericalJacobianOfStatePropagationInterface(x, data, dt)

    A = np.zeros([y.shape[0], x.shape[0]])

    for i in range(x.shape[0]):
        x[i] += ep
        y_i = numericalJacobianOfStatePropagationInterface(x, data, dt)
        A[i] = (y_i - y)/ep
        x[i] -= ep

    return A



# Sampling period
dt = .1
t_steps = 500
state = np.zeros(5)

state_hist = np.zeros([t_steps, 5])
imu_hist = np.zeros([t_steps, 2])


# Setup simulated data
for i in range(t_steps):
    # generate a rate to propagate states with
    accel = uniform(-10, 10)
    theta_dot = uniform(-0.2, 0.2)
    imu = np.array([accel, theta_dot])

    # propagating the state with the IMU measurement
    state = rk4(state, imu, ode, dt)

    # saving off the current state
    state_hist[i] = state *1.
    imu_hist[i] = imu*1.


# kf stuff
state = np.zeros([5])
cov = np.eye(5) * .001

kf_state_hist = np.zeros([t_steps, 5])
kf_cov_hist = np.zeros([t_steps, 5,5])
kf_meas_hist = np.zeros([t_steps, 3])
kf_imu_hist = np.zeros([t_steps, 2])

# imu accel and gyro noise
accel_cov = 1e-4
gyro_cov  = 1e-4
Q_imu = np.array([[.1, 0],[0, .01]])

r_meas = 1e-6

#  running the data through the KF with noised measurements
for i in range(t_steps):

    # propagating the state
    imu_meas = imu_hist[i]
    imu_meas[0] += np.random.randn(1)[0] * accel_cov**.5
    imu_meas[1] += np.random.randn(1)[0] * gyro_cov**.5

    A = numericalDifference(state, imu_meas, dt)
    cov = A.dot(cov.dot(A.T))

    ###
    # TODO : calculate how the accel and gyro noise turn into the process noise for the system
    ###
    # A_state_wrt_imu = jacobianOfPropagationWrtIMU
    # Q = A_state_wrt_imu * Q_imu * A_state_wrt_imu.T
    # cov += Q
    # sloppy placeholder
    cov += np.eye(5) * .1

    state = rk4(state, imu_meas, ode, dt)

    # measurement update
    zt = state[:3] + np.random.randn(1) *r_meas**.5
    zt_hat = state[:3]

    H = np.zeros([3,5])
    H[:3,:3] = np.eye(3)

    S = np.linalg.inv(H.dot(cov.dot(H.T)) + r_meas * np.eye(3))
    K = cov.dot(H.T).dot( S )

    state = state + K.dot(zt - zt_hat)
    cov = (np.eye(5) - K.dot(H)).dot(cov)

    kf_state_hist[i] = state
    kf_cov_hist[i] = cov
    kf_meas_hist[i] = zt_hat
    kf_imu_hist[i] = imu_meas



plt.plot(state_hist[:,0], state_hist[:,1], linewidth=3)
plt.plot(kf_state_hist[:,0], kf_state_hist[:,1], linewidth=3)
plt.legend(['Ground truth', 'kf est'])
plt.grid()
plt.show()
