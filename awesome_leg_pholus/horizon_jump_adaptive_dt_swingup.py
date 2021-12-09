##################### IMPORTS ########################

from horizon.problem import Problem
from horizon.utils import kin_dyn, utils, resampler_trajectory
import casadi_kin_dyn
from horizon.transcriptions import transcriptor
import numpy as np
from horizon.solvers import solver
import casadi as cs
from horizon.ros.replay_trajectory import *
from horizon.utils.plotter import PlotterHorizon
import matplotlib.pyplot as pyplt

###################### INITIALIZATION ########################

# "Loading the "URDF"
urdf_path = "/home/andreap/hhcm_ws/src/awesome_leg_pholus/description/urdf/awesome_leg_complete.urdf"
urdf = open(urdf_path, "r").read()
urdf_awesome_leg = casadi_kin_dyn.py3casadi_kin_dyn.CasadiKinDyn(urdf)

n_q = urdf_awesome_leg.nq()  # number of joints
n_v = urdf_awesome_leg.nv()  # number of dofs

# Setting some of the problem's parameters

n_nodes = 100
n_takeoff = 40  # node index at takeoff
n_touchdown = 80  # node index at touchdown

prb = Problem(n_nodes)  # initialization of a problem object

transcriptor_name = "multiple_shooting"  # other option: "direct_collocation"
trans_opt = dict(integrator="RK4")  # dictionary with the chosen integrator name

# Creating the state variables
q_p = prb.createStateVariable("q_p", n_q)
q_p_dot = prb.createStateVariable("q_p_dot",
                                  n_v)  # here q_p_dot is actually not the derivative of the lagrangian state vector

# Creating an additional input variable for the contact forces on the foot tip
f_contact = prb.createInputVariable("f_contact", 3)  # dimension 3
f_contact[2].setInitialGuess(100)  # initial guess (set to leg's weight)
f_contact[2].setLowerBounds(0)  # the vertical component of f_contact needs to be always positive
contact_map = dict(tip=f_contact)  # creating a contact map for applying the input to the foot

# initial joint config (ideally it would be given from measurements)
q_init = [0., 0., 0.]

###################### DEFINING BOUNDS ########################
q_p[0].setBounds(-0.4, 0.4)

tau_lim = np.array([0, 50, 50])  # effort limits (test_rig passive joint effort limit)
q_p.setBounds(q_init, q_init, 0)  # imposing the initial condition on q_p of the first node ("0")
q_p_dot.setBounds([0., 0., 0.], [0., 0., 0.], 0)  # zero initial "velocity"
q_p_ddot = prb.createInputVariable("q_p_ddot", n_v)  # using joint accelerations as an input variable
dt = prb.createInputVariable("dt", 1)  # using dt as an additional input
dt.setBounds(0.01, 0.1)  # bounds on dt

x, xdot = utils.double_integrator(q_p, q_p_dot, q_p_ddot)  # building the full state

prb.setDynamics(xdot)  # we are interested in the xdot
prb.setDt(dt)  # setting problem's dt

trscptr = transcriptor.Transcriptor.make_method(transcriptor_name, prb, trans_opt)  # setting the transcriptor

tau = kin_dyn.InverseDynamics(urdf_awesome_leg, contact_map.keys(),
                              casadi_kin_dyn.py3casadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q_p, q_p_dot,
                                                                                                      q_p_ddot,
                                                                                                      contact_map)  # obtaining the joint efforts

####################### DEFINING BOUNDS/CONSTRAINTS ###########################

fk = cs.Function.deserialize(urdf_awesome_leg.fk("hip1_1"))  # deserializing
position_LF_HIP_initial = fk(q=q_init)["ee_pos"]  # initial hip position (numerical)
position_LF_HIP = fk(q=q_p)["ee_pos"]  # hip position (symbolic)

# dfk_hip=cs.Function.deserialize(urdf_awesome_leg.frameVelocity("LF_HIP",casadi_kin_dyn.py3casadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
dfk_foot = cs.Function.deserialize(
    urdf_awesome_leg.frameVelocity("tip", casadi_kin_dyn.py3casadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
fk_foot = cs.Function.deserialize(urdf_awesome_leg.fk("tip"))

# v_LF_HIP=dfk_hip(q=q_p,qdot=q_p_dot)["ee_vel_linear"]
v_LF_FOOT = dfk_foot(q=q_p, qdot=q_p_dot)["ee_vel_linear"]  # foot vertical velocity
p_LF_FOOT = fk_foot(q=q_p)["ee_pos"]  # foot vertical position
p_LF_FOOT_init = fk_foot(q=q_init)["ee_pos"]  # foot vertical position

tau_cnstrnt = prb.createIntermediateConstraint("dynamics_feas", tau)  # dynamics feasibility constraint
tau_cnstrnt.setBounds(-tau_lim, tau_lim)  # setting input limits
# prb.createConstraint("hip_height", position_LF_HIP - position_LF_HIP_initial)
# prb.createConstraint("hip_vel", v_LF_HIP)
prb.createConstraint("foot_vel_bf_touchdown", v_LF_FOOT,
                     nodes=range(0, n_takeoff + 1))  # no vertical velocity of the foot before touchdown
prb.createConstraint("foot_vel_aftr_touchdown", v_LF_FOOT,
                     nodes=range(n_touchdown, n_nodes + 1))  # no vertical velocity of the foot after touchdown
prb.createConstraint("foot_pos_restoration", p_LF_FOOT - p_LF_FOOT_init,
                     nodes=n_nodes)  # restore foot position at the end of the optimization horizon
prb.createConstraint("GRF_zero", f_contact,
                     nodes=range(n_takeoff, n_touchdown))  # 0 vertical velocity during flight
prb.createFinalConstraint("final_joint_zero_vel", q_p_dot)  # joints are still at the end of the optimization horizon

############################# CREATING THE COST FUNCTION ######################################

weight_contact_cost = 1e-2  # minimizing the contact force
weight_postural_cost = 1000
weight_q_ddot = 0.5
weight_hip_height_jump = 200

prb.createIntermediateCost("min_f_contact", weight_contact_cost * cs.sumsqr(f_contact))
prb.createIntermediateCost("min_q_ddot", weight_q_ddot * cs.sumsqr(
    q_p_ddot))  # minimizing the joint accelerations ("responsiveness" of the trajectory)
prb.createFinalCost("postural", weight_postural_cost * cs.sumsqr(
    q_p - q_init))  # penalizing the difference between the initial position and the final one (using it as a constraint does not work)

prb.createIntermediateCost("max_hip_height_jump", weight_hip_height_jump * cs.sumsqr(1/(position_LF_HIP[2])),
                           nodes=range(n_takeoff, n_touchdown))

########################## SOLVER ##########################

slvr_opt = {"ipopt.tol": 1e-4, "ipopt.max_iter": 1000, "ipopt.linear_solver":"ma57"}
slvr = solver.Solver.make_solver("ipopt", prb, slvr_opt)

slvr.solve()  # solving
solution = slvr.getSolutionDict()
dt_opt = slvr.getDt()

joint_names = urdf_awesome_leg.joint_names()
joint_names.remove("universe")  # removing the "universe joint"

cnstr_opt = slvr.getConstraintSolutionDict()

# plt = PlotterHorizon(prb, solution)
# plt.plotVariables()
# plt.plotFunctions()

solution_GRF = solution["f_contact"]
solution_q_p = solution["q_p"]
solution_p_LF_FOOT = fk_foot(q=solution_q_p)["ee_pos"]  # foot vertical position

################### RESAMPLING (necessary because dt is variable) #####################à
q_sym = cs.SX.sym('q', n_q)
q_dot_sym = cs.SX.sym('q_dot', n_v)
q_ddot_sym = cs.SX.sym('q_ddot', n_v)
x, x_dot = utils.double_integrator(q_sym, q_dot_sym, q_ddot_sym)

dae = {'x': x, 'p': q_ddot_sym, 'ode': x_dot, 'quad': 1}

dt_res = 0.0001
sol_contact_map = dict(tip=solution["f_contact"])  # creating a contact map for applying the input to the foot

p_res, v_res, a_res, frame_res_force_mapping, tau_res = resampler_trajectory.resample_torques(solution["q_p"],
                                                                                              solution["q_p_dot"],
                                                                                              solution["q_p_ddot"],
                                                                                              solution["dt"].flatten(),
                                                                                              dt_res,
                                                                                              dae, sol_contact_map,
                                                                                              urdf_awesome_leg,
                                                                                              casadi_kin_dyn.py3casadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

rpl_traj = replay_trajectory(dt_res, joint_names, p_res)  # replaying the (resampled) trajectory

rpl_traj.sleep(1.0)
rpl_traj.replay(is_floating_base=False)


time_vector = np.zeros([n_nodes + 1])
time_vector_res = np.zeros([p_res.shape[1]])

for i in range(1, solution["dt"].shape[1]):
    time_vector[i] = time_vector[i - 1] + solution["dt"][:, i - 1]
for i in range(1, p_res.shape[1] - 1):
    time_vector_res[i] = time_vector_res[i - 1] + dt_res

pyplt.plot(time_vector[1:], solution_GRF[2])
pyplt.figure()
pyplt.plot(time_vector, np.squeeze(np.asarray(solution_p_LF_FOOT[2, :].toarray())))

########################## MANUAL PLOTS ##########################

pyplt.plot(time_vector_res[:-1], frame_res_force_mapping["tip"][0, :], label="GRF_x")
pyplt.plot(time_vector_res[:-1], frame_res_force_mapping["tip"][1, :], label="GRF_y")
pyplt.plot(time_vector_res[:-1], frame_res_force_mapping["tip"][2, :], label="GRF_z")
pyplt.legend(loc="upper left")
pyplt.title("Ground reaction forces", fontdict=None, loc='center')

pyplt.figure()
pyplt.plot(time_vector_res[:-1], p_res[0, :-1], label="q1")
pyplt.plot(time_vector_res[:-1], p_res[1, :-1], label="q2")
pyplt.plot(time_vector_res[:-1], p_res[2, :-1], label="q3")
pyplt.legend(loc="upper left")
pyplt.title("Joint states", fontdict=None, loc='center')

pyplt.figure()
pyplt.plot(time_vector_res[:-1], tau_res[1, :], label="hip_actuation")
pyplt.plot(time_vector_res[:-1], tau_res[2, :], label="knee_actuation")
pyplt.legend(loc="upper left")
pyplt.title("Joint efforts", fontdict=None, loc='center')

print(solution_p_LF_FOOT)
pyplt.figure()
pyplt.plot(time_vector_res[:-1], np.transpose(p_tip_res[2, :-1]), label="foot tip height")
pyplt.plot(time_vector_res[:-1], np.transpose(p_hip_res[2, :-1]), label="hip height")
pyplt.legend(loc="upper left")
pyplt.show()

