import torch

tensor_20n = torch.tensor([[-0.3944476, -0.25540486, 0.01000039, -0.19908206, 0.5572029,
                            0.5369497, 0.19070631, 0.48380834, -0.4328607, -0.20722768,
                            -0.02276316, -0.36313176, 0.4231819, 0.2779645, 0.3035667,
                            0.36547384, -0.15313634, 0.30386573, 0.48525652, 0.41800377],
                           [-0.42977545, -0.10574781, -0.3119562, 0.33579803, 0.06853759,
                            0.26304036, -0.2635372, 0.01790153, 0.12756805, -0.16413827,
                            -0.4119643, 0.56924146, 0.17398082, 0.29026505, 0.36825615,
                            0.56730866, 0.04020453, 0.1816869, -0.07828376, 0.5563348],
                           [0.14988348, 0.21509911, 0.08373737, 0.01071632, 0.46930093,
                            0.01413806, 0.39129218, 0.5055854, 0.21543477, 0.5805942,
                            -0.24861674, 0.22129522, -0.41882095, -0.19707926, 0.21793821,
                            0.3082475, -0.4321197, -0.3473076, -0.08767409, -0.4050355],
                           [-0.08813844, 0.55118364, -0.38116997, -0.33268303, -0.22755314,
                            -0.31138006, 0.06101719, 0.39177653, -0.42645153, -0.07265114,
                            0.21442789, 0.26522544, 0.5329072, 0.04049663, -0.3323004,
                            0.1908302, 0.55409884, 0.38335145, 0.54874796, -0.01837751],
                           [-0.08924848, 0.1322679, 0.30152267, 0.16165029, -0.12751588,
                            0.23724174, 0.04543222, 0.22703871, -0.04575785, 0.45298117,
                            0.1036612, 0.53421396, 0.57154745, 0.50517195, -0.0028578,
                            -0.24280263, 0.3371862, 0.247723, 0.541443, 0.27139637],
                           [0.00786474, -0.04168747, 0.4104852, 0.35246044, 0.56337965,
                            0.15341437, -0.3304049, -0.12762526, 0.32202527, 0.545145,
                            -0.21641718, -0.18699844, -0.2026263, -0.36954567, 0.4895186,
                            0.47220424, 0.09068766, 0.44405046, -0.20998344, 0.5075927],
                           [0.49095201, 0.41101494, 0.21398166, 0.21462555, -0.331933,
                            -0.12087005, 0.45160398, 0.24770725, 0.5413361, 0.266893,
                            0.2313734, 0.08403852, 0.15956117, 0.52387303, 0.4871623,
                            -0.27421394, -0.01535495, 0.43708712, 0.5973396, 0.28036603],
                           [-0.17102572, 0.37877804, 0.41810066, -0.3449817, -0.37525892,
                            -0.20568702, 0.2813632, -0.24619766, -0.43956336, 0.20004466,
                            -0.40867245, -0.36645463, 0.07017035, -0.32065222, 0.5769757,
                            0.42302382, -0.42399606, -0.12932019, -0.03260425, -0.20089523],
                           [0.55232215, 0.5946868, 0.51063555, 0.22719103, 0.5775366,
                            -0.2756641, 0.31141028, 0.31203932, 0.51279294, -0.44689354,
                            -0.06028134, -0.4386766, 0.17075372, 0.51601124, 0.06481452,
                            -0.09212375, 0.31110623, -0.24485414, -0.15690565, 0.23839103],
                           [0.01760931, 0.0803966, -0.08840767, 0.30297706, -0.08050612,
                            0.35970938, 0.5652488, -0.06407811, 0.24854122, 0.43263373,
                            0.23135488, -0.33536956, -0.01456508, 0.4138257, 0.40549135,
                            0.16205105, 0.24006209, 0.09235063, 0.06867236, 0.5763436],
                           [0.09905139, 0.5284948, 0.23995174, -0.0759218, -0.38261902,
                            -0.24110048, -0.00919148, -0.4391865, 0.37438473, -0.23594932,
                            -0.4222335, 0.47743, -0.00760848, 0.27191636, 0.4205516,
                            0.40840426, 0.2783853, 0.46006075, -0.152875, 0.02567722],
                           [-0.30275473, 0.3581606, 0.49172336, 0.47791097, 0.3926787,
                            -0.42432094, -0.06383026, 0.50023764, 0.01129873, 0.00557361,
                            0.46428883, 0.25438827, -0.35737455, -0.0214129, -0.22712846,
                            0.33437902, -0.2851146, -0.3872008, 0.22907753, -0.20413606],
                           [-0.29718113, 0.46220723, -0.21581762, 0.22314215, -0.30570796,
                            -0.42528176, 0.54818916, -0.11756258, -0.13900054, -0.3548733,
                            0.06409074, -0.01498313, -0.26009578, -0.08687681, -0.3999811,
                            0.51868814, -0.01359185, 0.29835775, 0.09458798, 0.53543276],
                           [0.4370323, 0.00988203, 0.1735845, -0.41337252, 0.36642265,
                            -0.31179887, -0.23165017, 0.25878257, 0.2011327, 0.4956968,
                            -0.08192985, 0.39770737, 0.3789402, -0.2111448, 0.44502768,
                            0.2884815, -0.25929, 0.5610071, 0.3833456, -0.06185273],
                           [0.5188865, -0.17089163, 0.48052567, -0.24654178, -0.31302956,
                            -0.06123647, -0.3699713, -0.07343723, 0.13688648, 0.42430958,
                            -0.32009456, 0.39881265, -0.2323855, 0.33441958, 0.55227304,
                            -0.43755066, 0.30460864, -0.14910737, 0.012185, 0.2363077],
                           [-0.4385534, 0.4572212, 0.5141367, -0.15397254, 0.5242115,
                            -0.236549, 0.40553087, -0.28150368, 0.15353632, 0.23127685,
                            0.180206, -0.39431283, 0.37391555, 0.51304907, 0.13399644,
                            -0.07374877, 0.3362726, 0.45573175, -0.30090636, 0.13750567],
                           [0.12926017, -0.04112987, -0.34534612, 0.5590492, -0.07573898,
                            -0.3000568, -0.09557287, -0.4222823, 0.56653595, 0.5801425,
                            -0.13607347, 0.16475667, 0.05208641, -0.2955727, -0.28930092,
                            0.5699783, 0.5037698, 0.45170972, 0.23013514, 0.5019224],
                           [0.4970048, 0.13768364, -0.09162921, -0.3825185, 0.21656366,
                            -0.25058132, 0.35400537, 0.3499934, 0.00483783, 0.06807683,
                            -0.13406503, 0.28789458, -0.41336378, 0.41399857, 0.40916958,
                            0.1068811, 0.22713546, 0.5411941, 0.00477971, 0.25375453],
                           [0.41855115, -0.36570048, 0.51661235, 0.3072519, -0.326462,
                            -0.21109556, 0.3843817, 0.13650133, -0.3975635, 0.38124403,
                            0.35658997, 0.14253527, -0.24175669, 0.34477696, 0.05850986,
                            0.23501284, -0.00593022, -0.29938683, -0.3913799, 0.02178242],
                           [0.06673021, 0.3554574, 0.2514149, 0.4506094, -0.03815171,
                            0.55634314, 0.37917733, -0.25555557, 0.5574373, -0.40197423,
                            0.55788594, 0.3493342, 0.31102636, 0.28095907, 0.5434031,
                            -0.43877873, 0.5608717, -0.3740805, -0.00118281, -0.25279808]])

tensor_25n = torch.tensor([[0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.0000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.0000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.0000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.0000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.0000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.0000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.0000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.0000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.0000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.0000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.0000, 0.5000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.0000, 0.5000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.0000, 0.5000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.0000, 0.5000],
                           [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.0000]])