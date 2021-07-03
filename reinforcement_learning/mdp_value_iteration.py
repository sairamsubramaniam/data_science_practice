
import copy
import numpy as np

np.set_printoptions(suppress=True)


# ==================== LEC17 Q2 =========================
# T = np.array([
#     [
#         [1/2,1/2,0,0,0],
#         [1/2,1/2,0,0,0],
#         [2/3,1/3,0,0,0]
#     ],
# 
#     [
#         [1/3,2/3,0,0,0],
#         [1/4,1/2,1/4,0,0],
#         [0,2/3,1/3,0,0]
#     ],
# 
#     [
#         [0,1/3,2/3,0,0],
#         [0,1/4,1/2,1/4,0],
#         [0,0,2/3,1/3,0]
#     ],
# 
#     [
#         [0,0,1/3,2/3,0],
#         [0,0,1/4,1/2,1/4],
#         [0,0,0,2/3,1/3]
#     ],
# 
#     [
#         [0,0,0,1/3,2/3],
#         [0,0,0,1/2,1/2],
#         [0,0,0,1/2,1/2]
#     ]
# ])

# R = np.zeros(T.shape)
# R[-1,-3:,:] = 1

# V = np.array([0,0,0,0,0])
# gamma = 0.5





# ======================= HW5 Tab1 Q5 ====================
# T = np.array([
#     [
#         [0,1,0,0],
#         [0,0,0,0],
#     ],
# 
#     [
#         [0,1,0,0],
#         [1,0,0,0],
#     ],
# 
#     [
#         [0,0,0,1],
#         [0,1,0,0],
#     ],
# 
#     [
#         [0,0,0,0],
#         [0,0,1,0],
#     ]
# ])

# R = np.array([[[1,1,1,1],[0,0,0,0]],
#               [[1,1,1,1],[1,1,1,1]],
#               [[10,10,10,10],[1,1,1,1]],
#               [[0,0,0,0],[10,10,10,10]]])

# V = np.array([0,0,0,10])
# gamma = 0.75






# ================= HW5 Tab2 =======================
# T = np.array([
#     [
#         [1,0,0,0,0,0],
#         [1,0,0,0,0,0],
#     ],
# 
#     [
#         [0,0.3,0,0.7,0,0],
#         [1,0,0,0,0,0],
#     ],
# 
#     [
#         [0,0,0.3,0,0.7,0],
#         [0,1,0,0,0,0],
#     ],
# 
#     [
#         [0,0,0,0.3,0,0.7],
#         [0,0,1,0,0,0],
#     ],
# 
#     [
#         [0,0,0,0,1,0],
#         [0,0,0,1,0,0],
#     ],
# 
#     [
#         [0,0,0,0,0,1],
#         [0,0,0,0,1,0],
#     ]
# ])


# def reward_func(s, s_prime):
#     if s != s_prime:
#         return abs(s_prime - s)**(1/3)
#     elif s != 0:
#         return (s+4)**(-1/2)
#     else:
#         return 0
# 
# S = range(6)
# A = range(2)
# V = np.zeros(len(S))
# R = np.zeros((len(S), 2, len(S)))
# gamma = 0.6
# 
# for s in S:
#     for a in A:
#         for s_prime in S:
#             R[s,a,s_prime] = reward_func(s, s_prime)



# =================== HW5 Tab3 =======================
# def t_func():
#     s = ["s1","s2", "s3", "s4",
#          "s5","s6", "s7", "s8",
#          "s9","s10","s11","s12"]
#     a = ["up","right","down","left"]
# 
#     t = np.zeros((len(s), len(a), len(s)))
# 
# 
#     # s1
#     t[s.index("s1"), a.index("right"), s.index("s2")] = 1
#     t[s.index("s1"), a.index("down"), s.index("s5")] = 1
# 
#     # s2
#     t[s.index("s2"), a.index("right"), s.index("s3")] = 1
#     t[s.index("s2"), a.index("left"), s.index("s1")] = 1
# 
#     # s3
#     t[s.index("s3"), a.index("right"), s.index("s4")] = 1
#     t[s.index("s3"), a.index("down"), s.index("s7")] = 1
#     t[s.index("s3"), a.index("left"), s.index("s2")] = 1
# 
#     # s4
#     t[s.index("s4"), a.index("down"), s.index("s8")] = 1
#     t[s.index("s4"), a.index("left"), s.index("s3")] = 1
# 
#     # s5
#     t[s.index("s5"), a.index("up"), s.index("s1")] = 1
#     t[s.index("s5"), a.index("down"), s.index("s9")] = 1
# 
#     # s6
#     t[s.index("s6"),:,:] = 0
# 
#     # s7
#     t[s.index("s7"), a.index("up"), s.index("s3")] = 1
#     t[s.index("s7"), a.index("right"), s.index("s8")] = 1
#     t[s.index("s7"), a.index("down"), s.index("s11")] = 1
# 
#     # s8
#     t[s.index("s8"), a.index("up"), s.index("s4")] = 1
#     t[s.index("s8"), a.index("down"), s.index("s12")] = 1
#     t[s.index("s8"), a.index("left"), s.index("s7")] = 1
# 
#     # s9
#     t[s.index("s9"), a.index("up"), s.index("s5")] = 1
#     t[s.index("s9"), a.index("right"), s.index("s10")] = 1
# 
#     # s10
#     t[s.index("s10"), a.index("right"), s.index("s11")] = 1
#     t[s.index("s10"), a.index("left"), s.index("s9")] = 1
# 
#     # s11
#     t[s.index("s11"), a.index("up"), s.index("s7")] = 1
#     t[s.index("s11"), a.index("right"), s.index("s12")] = 1
#     t[s.index("s11"), a.index("left"), s.index("s10")] = 1
# 
#     # s12
#     t[s.index("s12"), a.index("up"), s.index("s8")] = 1
#     t[s.index("s12"), a.index("left"), s.index("s11")] = 1
# 
# 
#     return t
# 
# 
# def r_func():
#     r = np.ones((12,4,12)) * -0.04
#     r[:,:,3] = 0.96
#     r[:,:,7] = -1.04
#     return r
# 
# 
# T = t_func()
# R = r_func()
# V = np.zeros(12)
# gamma = 1



# ==================== Project 5 Episodic Reward ========================
T = np.array([
    [
        [],
        [],
        [],
        []
    ],
    [
        [],
        [],
        [],
        []
    ],
    [
        [],
        [],
        [],
        []
    ],
    [
        [],
        [],
        [],
        []
    ]
])



# THE ALGORITHM USING NUMPY ARRAYS
for k in range(220):
    Q_sa = T * (R +  gamma * V[None,None,:])
    
    Q_s = Q_sa.sum(axis=2)
    Q_s[Q_s==0] = -(10**6)
    V = Q_s.max(axis=1)
    
    # V = Q_sa.sum(axis=2).max(axis=1)
    
    print()
    print("********************************")
    print()
    print("================ V ================")
    print(V)
    print("================ Q_s ================")
    print(Q_s)


# THE ALGORITHM USING FOR LOOPS
# def one_iter(T, R, old_V, gamma):
#     new_V = [0] * len(old_V)
#     for s in S:
#         Q = []
#         for a in A:
#             Q_sa = 0
#             for s_prime in S:
#                 t = T[s, a, s_prime]
#                 r = R[s, a, s_prime]
#                 Q_sa += t*(r + gamma*old_V[s_prime])
#             Q.append(Q_sa)
#         print("========= Q ==========")
#         print(Q)
#         new_V[s] = max(Q)
#     return new_V
# 
# 
# for k in range(10):
#     V = one_iter(T=T, R=R, old_V=V, gamma=gamma)
#     print("*****************************")
#     print(V)




