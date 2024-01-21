# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
""" Constants for Runge-Kutta Integration Methods.

Based off Scipy's implementation and references.
"""
cdef size_t i

########################################################################################################################
# Runge-Kutta 2(3)
########################################################################################################################
cdef size_t order_RK23       = 3
cdef size_t error_order_RK23 = 2
cdef size_t n_stages_RK23    = 3
cdef size_t A_rows_RK23      = 3
cdef size_t A_cols_RK23      = 3

cdef double[9] A_RK23
cdef double* A_RK23_ptr = &A_RK23[0]

cdef double[3] B_RK23
cdef double* B_RK23_ptr = &B_RK23[0]

cdef double[3] C_RK23
cdef double* C_RK23_ptr = &C_RK23[0]

cdef double[4] E_RK23
cdef double* E_RK23_ptr = &E_RK23[0]
# A - Row 0
A_RK23_ptr[0] = 0.
A_RK23_ptr[1] = 0.
A_RK23_ptr[2] = 0.
# A - Row 1
A_RK23_ptr[3] = 1. / 2.
A_RK23_ptr[4] = 0.
A_RK23_ptr[5] = 0.
# A - Row 2
A_RK23_ptr[6] = 0.
A_RK23_ptr[7] = 3. / 4.
A_RK23_ptr[8] = 0.

# B pointer
B_RK23_ptr[0] = 2. / 9.
B_RK23_ptr[1] = 1. / 3.
B_RK23_ptr[2] = 4. / 9.

# C Pointer
C_RK23_ptr[0] = 0.
C_RK23_ptr[1] = 1. / 2.
C_RK23_ptr[2] = 3. / 4.

# E Pointer
E_RK23_ptr[0] = 5. / 72.
E_RK23_ptr[1] = -1. / 12.
E_RK23_ptr[2] = -1. / 9.
E_RK23_ptr[3] = 1. / 8.


########################################################################################################################
# Runge-Kutta 4(5)
########################################################################################################################
cdef size_t order_RK45       = 5
cdef size_t error_order_RK45 = 4
cdef size_t n_stages_RK45    = 6
cdef size_t A_rows_RK45      = 6
cdef size_t A_cols_RK45      = 5

cdef double[30] A_RK45
cdef double* A_RK45_ptr = &A_RK45[0]

cdef double[6] B_RK45
cdef double* B_RK45_ptr = &B_RK45[0]

cdef double[6] C_RK45
cdef double* C_RK45_ptr = &C_RK45[0]

cdef double[7] E_RK45
cdef double* E_RK45_ptr = &E_RK45[0]
# A - Row 0
A_RK45_ptr[0] = 0.
A_RK45_ptr[1] = 0.
A_RK45_ptr[2] = 0.
A_RK45_ptr[3] = 0.
A_RK45_ptr[4] = 0.
# A - Row 1
A_RK45_ptr[5] = 1. / 5.
A_RK45_ptr[6] = 0.
A_RK45_ptr[7] = 0.
A_RK45_ptr[8] = 0.
A_RK45_ptr[9] = 0.
# A - Row 2
A_RK45_ptr[10] = 3. / 40.
A_RK45_ptr[11] = 9. / 40.
A_RK45_ptr[12] = 0.
A_RK45_ptr[13] = 0.
A_RK45_ptr[14] = 0.
# A - Row 3
A_RK45_ptr[15] = 44. / 45.
A_RK45_ptr[16] = -56. / 15.
A_RK45_ptr[17] = 32. / 9.
A_RK45_ptr[18] = 0.
A_RK45_ptr[19] = 0.
# A - Row 4
A_RK45_ptr[20] = 19372. / 6561.
A_RK45_ptr[21] = -25360. / 2187.
A_RK45_ptr[22] = 64448. / 6561.
A_RK45_ptr[23] = -212. / 729.
A_RK45_ptr[24] = 0.
# A - Row 5
A_RK45_ptr[25] = 9017. / 3168.
A_RK45_ptr[26] = -355. / 33.
A_RK45_ptr[27] = 46732. / 5247.
A_RK45_ptr[28] = 49. / 176.
A_RK45_ptr[29] = -5103. / 18656.

# B pointer
B_RK45_ptr[0] = 35. / 384.
B_RK45_ptr[1] = 0.
B_RK45_ptr[2] = 500. / 1113.
B_RK45_ptr[3] = 125. / 192.
B_RK45_ptr[4] = -2187. / 6784.
B_RK45_ptr[5] = 11. / 84.

# C Pointer
C_RK45_ptr[0] = 0.
C_RK45_ptr[1] = 1. / 5.
C_RK45_ptr[2] = 3. / 10.
C_RK45_ptr[3] = 4. / 5.
C_RK45_ptr[4] = 8. / 9.
C_RK45_ptr[5] = 1.

# E Pointer
E_RK45_ptr[0] = -71. / 57600.
E_RK45_ptr[1] = 0.
E_RK45_ptr[2] = 71. / 16695.
E_RK45_ptr[3] = -71. / 1920.
E_RK45_ptr[4] = 17253. / 339200.
E_RK45_ptr[5] = -22. / 525.
E_RK45_ptr[6] = 1. / 40.


########################################################################################################################
# Runge-Kutta DOP 8(5; 3)
########################################################################################################################
cdef size_t order_DOP853       = 8
cdef size_t error_order_DOP853 = 7
cdef size_t n_stages_DOP853    = 12
cdef size_t A_rows_DOP853      = 12
cdef size_t A_cols_DOP853      = 12

# Note both A and C are the _reduced_ versions. The full A and C are not shown.
cdef double[144] A_DOP853
cdef double* A_DOP853_ptr = &A_DOP853[0]

cdef double[12] B_DOP853
cdef double* B_DOP853_ptr = &B_DOP853[0]

cdef double[12] C_DOP853
cdef double* C_DOP853_ptr = &C_DOP853[0]

cdef double[13] E3_DOP853
cdef double* E3_DOP853_ptr = &E3_DOP853[0]
cdef double[13] E5_DOP853
cdef double* E5_DOP853_ptr = &E5_DOP853[0]
# A - Row 0
A_DOP853_ptr[0] = 0.
A_DOP853_ptr[1] = 0.
A_DOP853_ptr[2] = 0.
A_DOP853_ptr[3] = 0.
A_DOP853_ptr[4] = 0.
A_DOP853_ptr[5] = 0.
A_DOP853_ptr[6] = 0.
A_DOP853_ptr[7] = 0.
A_DOP853_ptr[8] = 0.
A_DOP853_ptr[9] = 0.
A_DOP853_ptr[10] = 0.
A_DOP853_ptr[11] = 0.
# A - Row 1
A_DOP853_ptr[12] = 5.26001519587677318785587544488e-2
A_DOP853_ptr[13] = 0.
A_DOP853_ptr[14] = 0.
A_DOP853_ptr[15] = 0.
A_DOP853_ptr[16] = 0.
A_DOP853_ptr[17] = 0.
A_DOP853_ptr[18] = 0.
A_DOP853_ptr[19] = 0.
A_DOP853_ptr[20] = 0.
A_DOP853_ptr[21] = 0.
A_DOP853_ptr[22] = 0.
A_DOP853_ptr[23] = 0.
# A - Row 2
A_DOP853_ptr[24] = 1.97250569845378994544595329183e-2
A_DOP853_ptr[25] = 5.91751709536136983633785987549e-2
A_DOP853_ptr[26] = 0.
A_DOP853_ptr[27] = 0.
A_DOP853_ptr[28] = 0.
A_DOP853_ptr[29] = 0.
A_DOP853_ptr[30] = 0.
A_DOP853_ptr[31] = 0.
A_DOP853_ptr[32] = 0.
A_DOP853_ptr[33] = 0.
A_DOP853_ptr[34] = 0.
A_DOP853_ptr[35] = 0.
# A - Row 3
A_DOP853_ptr[36] = 2.95875854768068491816892993775e-2
A_DOP853_ptr[37] = 0.
A_DOP853_ptr[38] = 8.87627564304205475450678981324e-2
A_DOP853_ptr[39] = 0.
A_DOP853_ptr[40] = 0.
A_DOP853_ptr[41] = 0.
A_DOP853_ptr[42] = 0.
A_DOP853_ptr[43] = 0.
A_DOP853_ptr[44] = 0.
A_DOP853_ptr[45] = 0.
A_DOP853_ptr[46] = 0.
A_DOP853_ptr[47] = 0.
# A - Row 4
A_DOP853_ptr[48] = 2.41365134159266685502369798665e-1
A_DOP853_ptr[49] = 0.
A_DOP853_ptr[50] = -8.84549479328286085344864962717e-1
A_DOP853_ptr[51] = 9.24834003261792003115737966543e-1
A_DOP853_ptr[52] = 0.
A_DOP853_ptr[53] = 0.
A_DOP853_ptr[54] = 0.
A_DOP853_ptr[55] = 0.
A_DOP853_ptr[56] = 0.
A_DOP853_ptr[57] = 0.
A_DOP853_ptr[58] = 0.
A_DOP853_ptr[59] = 0.
# A - Row 5
A_DOP853_ptr[60] = 3.7037037037037037037037037037e-2
A_DOP853_ptr[61] = 0.
A_DOP853_ptr[62] = 0.
A_DOP853_ptr[63] = 1.70828608729473871279604482173e-1
A_DOP853_ptr[64] = 1.25467687566822425016691814123e-1
A_DOP853_ptr[65] = 0.
A_DOP853_ptr[66] = 0.
A_DOP853_ptr[67] = 0.
A_DOP853_ptr[68] = 0.
A_DOP853_ptr[69] = 0. # # Nice
A_DOP853_ptr[70] = 0.
A_DOP853_ptr[71] = 0.
# A - Row 6
A_DOP853_ptr[72] = 3.7109375e-2
A_DOP853_ptr[73] = 0.
A_DOP853_ptr[74] = 0.
A_DOP853_ptr[75] = 1.70252211019544039314978060272e-1
A_DOP853_ptr[76] = 6.02165389804559606850219397283e-2
A_DOP853_ptr[77] = -1.7578125e-2
A_DOP853_ptr[78] = 0.
A_DOP853_ptr[79] = 0.
A_DOP853_ptr[80] = 0.
A_DOP853_ptr[81] = 0.
A_DOP853_ptr[82] = 0.
A_DOP853_ptr[83] = 0.
# A - Row 7
A_DOP853_ptr[84] = 3.70920001185047927108779319836e-2
A_DOP853_ptr[85] = 0.
A_DOP853_ptr[86] = 0.
A_DOP853_ptr[87] = 1.70383925712239993810214054705e-1
A_DOP853_ptr[88] = 1.07262030446373284651809199168e-1
A_DOP853_ptr[89] = -1.53194377486244017527936158236e-2
A_DOP853_ptr[90] = 8.27378916381402288758473766002e-3
A_DOP853_ptr[91] = 0.
A_DOP853_ptr[92] = 0.
A_DOP853_ptr[93] = 0.
A_DOP853_ptr[94] = 0.
A_DOP853_ptr[95] = 0.
# A - Row 8
A_DOP853_ptr[96] = 6.24110958716075717114429577812e-1
A_DOP853_ptr[97] = 0.
A_DOP853_ptr[98] = 0.
A_DOP853_ptr[99] = -3.36089262944694129406857109825
A_DOP853_ptr[100] = -8.68219346841726006818189891453e-1
A_DOP853_ptr[101] = 2.75920996994467083049415600797e1
A_DOP853_ptr[102] = 2.01540675504778934086186788979e1
A_DOP853_ptr[103] = -4.34898841810699588477366255144e1
A_DOP853_ptr[104] = 0.
A_DOP853_ptr[105] = 0.
A_DOP853_ptr[106] = 0.
A_DOP853_ptr[107] = 0.
# A - Row 9
A_DOP853_ptr[108] = 4.77662536438264365890433908527e-1
A_DOP853_ptr[109] = 0.
A_DOP853_ptr[110] = 0.
A_DOP853_ptr[111] = -2.48811461997166764192642586468
A_DOP853_ptr[112] = -5.90290826836842996371446475743e-1
A_DOP853_ptr[113] = 2.12300514481811942347288949897e1
A_DOP853_ptr[114] = 1.52792336328824235832596922938e1
A_DOP853_ptr[115] = -3.32882109689848629194453265587e1
A_DOP853_ptr[116] = -2.03312017085086261358222928593e-2
A_DOP853_ptr[117] = 0.
A_DOP853_ptr[118] = 0.
A_DOP853_ptr[119] = 0.
# A - Row 10
A_DOP853_ptr[120] = -9.3714243008598732571704021658e-1
A_DOP853_ptr[121] = 0.
A_DOP853_ptr[122] = 0.
A_DOP853_ptr[123] = 5.18637242884406370830023853209
A_DOP853_ptr[124] = 1.09143734899672957818500254654
A_DOP853_ptr[125] = -8.14978701074692612513997267357
A_DOP853_ptr[126] = -1.85200656599969598641566180701e1
A_DOP853_ptr[127] = 2.27394870993505042818970056734e1
A_DOP853_ptr[128] = 2.49360555267965238987089396762
A_DOP853_ptr[129] = -3.0467644718982195003823669022
A_DOP853_ptr[130] = 0.
A_DOP853_ptr[131] = 0.
# A - Row 11
A_DOP853_ptr[132] = 2.27331014751653820792359768449
A_DOP853_ptr[133] = 0.
A_DOP853_ptr[134] = 0.
A_DOP853_ptr[135] = -1.05344954667372501984066689879e1
A_DOP853_ptr[136] = -2.00087205822486249909675718444
A_DOP853_ptr[137] = -1.79589318631187989172765950534e1
A_DOP853_ptr[138] = 2.79488845294199600508499808837e1
A_DOP853_ptr[139] = -2.85899827713502369474065508674
A_DOP853_ptr[140] = -8.87285693353062954433549289258
A_DOP853_ptr[141] = 1.23605671757943030647266201528e1
A_DOP853_ptr[142] = 6.43392746015763530355970484046e-1
A_DOP853_ptr[143] = 0.

# B pointer
# Note: B is equal to the 13th row of the expanded version of A (which we do not define above)
B_DOP853_ptr[0] = 5.42937341165687622380535766363e-2
B_DOP853_ptr[1] = 0.
B_DOP853_ptr[2] = 0.
B_DOP853_ptr[3] = 0.
B_DOP853_ptr[4] = 0.
B_DOP853_ptr[5] = 4.45031289275240888144113950566
B_DOP853_ptr[6] = 1.89151789931450038304281599044
B_DOP853_ptr[7] = -5.8012039600105847814672114227
B_DOP853_ptr[8] = 3.1116436695781989440891606237e-1
B_DOP853_ptr[9] = -1.52160949662516078556178806805e-1
B_DOP853_ptr[10] = 2.01365400804030348374776537501e-1
B_DOP853_ptr[11] = 4.47106157277725905176885569043e-2

# C Pointer
# Note this is the reduced C array. The expanded version is not shown.
C_DOP853_ptr[0] = 0.
C_DOP853_ptr[1] = 0.526001519587677318785587544488e-01
C_DOP853_ptr[2] = 0.789002279381515978178381316732e-01
C_DOP853_ptr[3] = 0.118350341907227396726757197510
C_DOP853_ptr[4] = 0.281649658092772603273242802490
C_DOP853_ptr[5] = 0.333333333333333333333333333333
C_DOP853_ptr[6] = 0.25
C_DOP853_ptr[7] = 0.307692307692307692307692307692
C_DOP853_ptr[8] = 0.651282051282051282051282051282
C_DOP853_ptr[9] = 0.6
C_DOP853_ptr[10] = 0.857142857142857142857142857142
C_DOP853_ptr[11] = 1.0

# E3 Pointer
for i in range(13):
    if i == 12:
        # All except last value equals B (B length is one less than E3).
        E3_DOP853_ptr[i] = 0.
    else:
        E3_DOP853_ptr[i] = B_DOP853_ptr[i]
E3_DOP853_ptr[0] -= 0.244094488188976377952755905512
E3_DOP853_ptr[8] -= 0.733846688281611857341361741547
E3_DOP853_ptr[11] -= 0.220588235294117647058823529412e-1

# E5 Pointer
E5_DOP853_ptr[0] = 0.1312004499419488073250102996e-1
E5_DOP853_ptr[1] = 0.
E5_DOP853_ptr[2] = 0.
E5_DOP853_ptr[3] = 0.
E5_DOP853_ptr[4] = 0.
E5_DOP853_ptr[5] = -0.1225156446376204440720569753e+1
E5_DOP853_ptr[6] = -0.4957589496572501915214079952
E5_DOP853_ptr[7] = 0.1664377182454986536961530415e+1
E5_DOP853_ptr[8] = -0.3503288487499736816886487290
E5_DOP853_ptr[9] = 0.3341791187130174790297318841
E5_DOP853_ptr[10] = 0.8192320648511571246570742613e-1
E5_DOP853_ptr[11] = -0.2235530786388629525884427845e-1
E5_DOP853_ptr[12] = 0.
