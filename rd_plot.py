#! /Users/gcwhiteshadow/anaconda3/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


class RD_Curve:
    def __init__(self, index=['PSNR', 'MS-SSIM', 'bpp']):
        assert len(index) != 0
        assert any('PSNR' in s for s in index)
        assert any('MS-SSIM' in s for s in index)
        assert any('bpp' in s for s in index)

        self.index = index
        self.points = np.empty((0, len(self.index)))

        '''
        PSNR | MS-SSIM | bpp | ext....
        ===========================...
             |         |     |     ...
             |         |     |     ...
             |         |     |     ...
             |         |     |     ...
        '''

    def add_points(self, points: list):
        points_np = np.array(points)
        assert ((len(points_np.shape) == 1 and points_np.shape[0] == len(self.index)) or
                (len(points_np.shape) == 2 and points_np.shape[1] == len(self.index)))

        if len(points_np.shape) == 1:
            points_np = np.expand_dims(points_np, 0)

        '''
        [   [psnr_1, ms_ssim_1, bpp_1, ext.....], 
            .
            .
            .    
        ]
        '''

        self.points = np.concatenate([self.points, points_np], axis=0)

    def get_series(self, index_name: str):
        assert any(index_name in s for s in self.index)
        dict_name = {self.index[i]: i for i in range(0, len(self.index))}

        return self.points[:, dict_name[index_name]]

    @property
    def PSNR(self):
        return self.get_series('PSNR')

    @property
    def MS_SSIM(self):
        return self.get_series('MS-SSIM')

    @property
    def bpp(self):
        return self.get_series('bpp')


class RD_Plot:
    def __init__(self, metric='PSNR'):
        assert metric == 'PSNR' or metric == 'MS-SSIM'
        plt.figure(figsize=(12, 9))

        self.curves = []
        self.ax = plt.subplot(111)

        plt.title('{} on Kodak'.format(metric))
        plt.xlabel("bit-per-pixel")
        plt.ylabel(metric)

        xmajor = MultipleLocator(0.5)
        ymajor = MultipleLocator(5)

        self.ax.xaxis.set_major_locator(xmajor)
        self.ax.yaxis.set_major_locator(ymajor)

        xminor = MultipleLocator(0.1)
        yminor = MultipleLocator(1)

        self.ax.xaxis.set_minor_locator(xminor)
        self.ax.yaxis.set_minor_locator(yminor)

        plt.grid(b=True, which='major', color='black')
        plt.grid(b=True, which='minor', color='gray', linestyle='--')

    @staticmethod
    def add_curve(series_A: np.ndarray, series_B: np.ndarray, label='str',
                  color=None, style=None, width=None, marker=None):
        plt.plot(series_A, series_B, label=label, c=color, ls=style, lw=width, marker=marker)
        plt.legend(loc='lower right')

    def plot(self):
        plt.show()

    def save_figure(self, filename: str):
        plt.savefig(filename)


BPG = RD_Curve()
BPG.add_points([
    [23.787652, None, 0.023857],
    [24.169736, None, 0.02854],
    [24.583862, None, 0.034282],
    [25.005823, None, 0.041183],
    [25.441511, None, 0.049301],
    [25.892665, None, 0.058861],
    [26.341951, None, 0.070444],
    [26.815071, None, 0.084175],
    [27.298468, None, 0.100255],
    [27.80339, None, 0.119211],
    [28.299163, None, 0.140076],
    [28.836366, None, 0.165529],
    [29.370955, None, 0.193939],
    [29.927924, None, 0.226882],
    [30.519758, None, 0.264902],
    [31.114314, None, 0.308004],
    [31.707365, None, 0.353821],
    [32.333445, None, 0.406663],
    [32.953625, None, 0.465174],
    [33.589638, None, 0.528513],
    [34.260077, None, 0.602615],
    [34.921675, None, 0.681227],
    [35.560822, None, 0.76315],
    [36.239184, None, 0.855122],
    [36.892823, None, 0.954195],
    [37.539114, None, 1.058729],
    [38.225035, None, 1.178031],
    [38.89222, None, 1.302895],
    [39.508856, None, 1.427853],
    [40.150529, None, 1.570484],
    # [40.780304, None, 1.72431],
    # [41.391531, None, 1.890798],
    # [42.05465, None, 2.08568],
    # [42.720115, None, 2.296926],
    # [43.3373, None, 2.513738],
    # [43.993691, None, 2.762745],
    # [44.653016, None, 3.021654],
    # [45.30127, None, 3.311613],
    # [45.915667, None, 3.616902],
    # [46.518668, None, 3.94321],
])

Google_inference = RD_Curve()
Google_inference.add_points([
    [40.8280, None, 1.7386],
    [36.8613, None, 0.9648],
    [32.7323, None, 0.4814],
    [29.0628, None, 0.2064],
    [27.2412, None, 0.1275]
])

Google_reRun = RD_Curve()
Google_reRun.add_points([
    [38.7326, 0.9929, 1.8113],  # 0122_1019
    [36.6219, 0.9893, 1.2611],  # 0122_1020
    [36.0186, 0.9893, 1.1364],  # 0121_0324
    [31.9183, 0.9700, 0.4729],  # 0121_0023
    [29.8161, 0.9515, 0.2972],  # 0121_0325
    [25.9059, 0.8856, 0.0902]   # 0121_0024
])

Google_inference = RD_Curve()
Google_inference.add_points([
    [40.8280, None, 1.7386],
    [36.8613, None, 0.9648],
    [32.7323, None, 0.4814],
    [29.0628, None, 0.2064],
    [27.2412, None, 0.1275]
])

# 2FM lambda=30, 100, 300, 1000
FM2_wPCA_wRE = RD_Curve()
FM2_wPCA_wRE.add_points([
    [22.8404, 0.7885, 0.03349],
    [24.9357, 0.8249, 0.05771],
    [26.1662, 0.9051, 0.08631],
    [26.4656, None, 0.10367]
])

# 4FM lambda=30, 100, 200, 300, 3000
FM4_wPCA_wRE = RD_Curve()
FM4_wPCA_wRE.add_points([
    [23.5611, 0, 0.0556],
    [26.3808, 0, 0.1023],
    [27.0044, 0, 0.118],
    [27.6546, 0, 0.1446],
    [27.9303, 0, 0.1729]
])

# 8FM lambda=30, 100, 200, 300, 3000
FM8_wPCA_wRE = RD_Curve()
FM8_wPCA_wRE.add_points([
    [25.1125, 0.8733, 0.1054],
    [27.8471, 0.9307, 0.18655],
    [28.3093, 0.9383, 0.2068],
    [28.885, 0.9488, 0.2434],
    [29.3374, 0.9563, 0.289]
])

ETHZ_CVPR18_MSSSIM = RD_Curve()
ETHZ_CVPR18_MSSSIM.add_points([
    [None, 0.9289356, 0.1265306],
    [None, 0.9417454, 0.1530612],
    [None, 0.9497924, 0.1795918],
    [None, 0.9553684, 0.2061224],
    [None, 0.9598574, 0.2326531],
    [None, 0.9636625, 0.2591837],
    [None, 0.9668663, 0.2857143],
    [None, 0.9695684, 0.3122449],
    [None, 0.9718446, 0.3387755],
    [None, 0.9738012, 0.3653061],
    [None, 0.9755308, 0.3918367],
    [None, 0.9770696, 0.4183673],
    [None, 0.9784622, 0.4448980],
    [None, 0.9797252, 0.4714286],
    [None, 0.9808753, 0.4979592],
    [None, 0.9819255, 0.5244898],
    [None, 0.9828875, 0.5510204],
    [None, 0.9837722, 0.5775510],
    [None, 0.9845877, 0.6040816],
    [None, 0.9853407, 0.6306122],
    [None, 0.9860362, 0.6571429],
    [None, 0.9866768, 0.6836735],
    [None, 0.9872690, 0.7102041],
    [None, 0.9878184, 0.7367347],
    [None, 0.9883268, 0.7632653],
    [None, 0.9887977, 0.7897959],
    [None, 0.9892346, 0.8163265],
    [None, 0.9896379, 0.8428571]
])

# Google_pytorch = RD_Curve()
# Google_pytorch.add_points([
#     [36.2842, 0.9896, 1.1643],  # 0415_1337
#     [32.0858, 0.9694, 0.5137],  # 0415_1339
#     [30.1185, 0.9500, 0.3266],  # 0415_1340
#     [26.2197, 0.8810, 0.1015],  # 0415_1341
# ])

Google_pytorch = RD_Curve()
Google_pytorch.add_points([
    [36.4098, 0.9896, 1.0923],  # 0810_1622
    [32.1168, 0.9687, 0.4691],  # 0810_1623
    [30.1392, 0.9492, 0.2959],  # 0810_1624
    [26.2420, 0.8793, 0.0981],  # 0810_1625
])

Google_pytorch_3 = RD_Curve()
Google_pytorch_3.add_points([
    [38.61, None, 1.45],
    [36.8, None, 1.11],
    [32.3, None, 0.5],
    [30.1, None, 0.3]
])

Google_OctConv = RD_Curve()
Google_OctConv.add_points([
    [33.3737, 0.9869, 0.9080],
    [31.2465, 0.9691, 0.4327],
    [29.5892, 0.9527, 0.2910],
    [25.5575, 0.8833, 0.0977]
])

Google_OctConv_high = RD_Curve()
Google_OctConv_high.add_points([
    [38.3478, 0.9933, 1.5984],
    [36.5839, 0.9898, 1.1463],
])

CConv = RD_Curve()  # 0416_2324
CConv.add_points([
    [37.0171, 0.9919, 1.4643],
    [36.0083, 0.9890, 1.1856],
    [32.2798, 0.9717, 0.5965],
    [30.2597, 0.9550, 0.4059],
    [26.1214, 0.8950, 0.1372],
])

CConv_bypass = RD_Curve()
CConv_bypass.add_points([
    [36.8796, 0.9915, 1.4738],  # 0421_0622
    [35.8592, 0.9884, 1.1612],
    [32.2691, 0.9698, 0.5704],
    [30.3811, 0.9528, 0.3710],
    [26.2928, 0.8786, 0.1211],
])

CConv_bypass_high = RD_Curve()
CConv_bypass_high.add_points([
    [38.5634, 0.9925, 1.6371],  # 0520_2117
    [36.7143, 0.9887, 1.2293],
    [31.8376, 0.9690, 0.5462],
    [29.8565, 0.9519, 0.3570],
    [26.1155, 0.8871, 0.1247],
])

CConv_masked = RD_Curve()
CConv_masked.add_points([
    [36.9117, 0.9917, 1.4488],
    [35.9630, 0.9886, 1.1336],
    [32.3266, 0.9703, 0.5292],
    [30.4708, 0.9537, 0.3406],
    [26.4348, 0.8825, 0.1151],
])

CConv_high = RD_Curve()
CConv_high.add_points([
    [38.4323, 0.9922, 1.6303],  # 0519_2205
    [36.8512, 0.9885, 1.2606],
    [32.2268, 0.9661, 0.5615],
    [30.2850, 0.9466, 0.3687],
    [26.0127, 0.8551, 0.1299],
])

CConv_masked_high = RD_Curve()
CConv_masked_high.add_points([
    [38.4207, 0.9924, 1.5732],  # 0517_2059
    [36.6578, 0.9885, 1.1823],
    [32.0028, 0.9677, 0.5036],
    [30.1523, 0.9507, 0.3249],
    [26.3112, 0.8794, 0.1080],
])

CConv_masked_high_share1 = RD_Curve()
CConv_masked_high_share1.add_points([
    [38.4220, 0.9923, 1.5515],
    [36.7117, 0.9886, 1.1562],
    [32.2746, 0.9695, 0.5091],
    [30.3469, 0.9529, 0.3349],
    [25.9086, 0.8728, 0.1143],
])

CConv_masked_high_share2 = RD_Curve()
CConv_masked_high_share2.add_points([
    [38.7385, 0.9926, 1.5900],
    [37.0211, 0.9893, 1.2059],
    [32.7119, 0.9730, 0.5416],
    [30.8440, 0.9586, 0.3622],
    [26.6769, 0.8964, 0.1280],
])

CConv_masked_high_share3 = RD_Curve()
CConv_masked_high_share3.add_points([
    [38.4995, 0.9922, 1.4982],
    [35.9858, 0.9879, 1.1543],
    [31.8576, 0.9711, 0.5221],
    [29.9341, 0.9564, 0.3553],
    [25.6051, 0.8883, 0.1319],
])

CConv_masked_high_imp = RD_Curve()
CConv_masked_high_imp.add_points([
    [37.6259, 0.9851, 1.4555],  # 0518_2332
    [35.7534, 0.9813, 1.0876],
    [31.2694, 0.9623, 0.4555],
    [29.3696, 0.9424, 0.2914],
    [25.7359, 0.8673, 0.0953],
])

CConv_masked_high_fixed_imp = RD_Curve()
CConv_masked_high_fixed_imp.add_points([
    [38.3839, 0.9924, 1.3386],  # 0519_0944
    [36.7544, 0.9889, 0.9923],
    [32.2949, 0.9715, 0.3884],
    [30.5572, 0.9587, 0.2332],
    [26.7027, 0.9018, 0.0613],
])

CConv_bypass_high_imp = RD_Curve()
CConv_bypass_high_imp.add_points([
    [37.5466, 0.9897, 1.4690],  # 0521_0257
    [35.6379, 0.9843, 1.1019],
    [30.8429, 0.9566, 0.4715],
    [28.9626, 0.9339, 0.3033],
    [25.2385, 0.8506, 0.0989],
])

CConv_bypass_high_imp_end_to_end = RD_Curve()
CConv_bypass_high_imp_end_to_end.add_points([
    [37.7689, 0.9884, 2.0298],  # 0525_1221
    [35.8900, 0.9822, 1.5947],
    [30.1256, 0.9437, 0.6514],
    [27.8031, 0.9108, 0.3735],
    [24.5158, 0.8320, 0.0965],
])

CConv_bypass_high_fixed_imp = RD_Curve()
CConv_bypass_high_fixed_imp.add_points([
    [38.2401, 0.9907, 1.5028],  # 0521_0357
    [36.3817, 0.9861, 1.1308],
    [31.8286, 0.9632, 0.5075],
    [29.9731, 0.9447, 0.3308],
    [26.0990, 0.8707, 0.1066],
])

CConv_bypass_high_ld_1e_1 = RD_Curve()
CConv_bypass_high_ld_1e_1.add_points([
    # [39.7801, 0.9945, 2.2497],
    # [39.0015, 0.9933, 1.8033],
    # [38.0918, 0.9917, 1.5100],
    # [37.0678, 0.9894, 1.3201],
    # [35.9584, 0.9865, 1.1958],
    # [34.7925, 0.9825, 1.1146]
    [39.6108, 0.9940, 1.7362],
    [38.8654, 0.9930, 1.5021],
    [38.0194, 0.9916, 1.3463],
    [37.0871, 0.9897, 1.2426],
    [36.0843, 0.9873, 1.1741],
    [35.0395, 0.9841, 1.1288]
])

CConv_bypass_high_ld_5e_2 = RD_Curve()
CConv_bypass_high_ld_5e_2.add_points([
    # [38.0062, 0.9916, 1.7352],
    # [37.1801, 0.9899, 1.3661],
    # [36.2207, 0.9874, 1.1238],
    # [35.1553, 0.9841, 0.9653],
    # [34.0173, 0.9797, 0.8620],
    # [32.8519, 0.9738, 0.7951]
    [37.4891, 0.9907, 1.2798],
    [36.6956, 0.9890, 1.0997],
    [35.8239, 0.9869, 0.9800],
    [34.8918, 0.9840, 0.9005],
    [33.9157, 0.9804, 0.8481],
    [32.9318, 0.9757, 0.8138]
])

CConv_bypass_high_ld_1e_2 = RD_Curve()
CConv_bypass_high_ld_1e_2.add_points([
    # [32.9659, 0.9759, 0.8390],
    # [32.2401, 0.9717, 0.6239],
    # [31.4125, 0.9659, 0.4860],
    # [30.5129, 0.9583, 0.3984],
    # [29.5684, 0.9483, 0.3440],
    # [28.6288, 0.9357, 0.3109]
    [32.5407, 0.9738, 0.5731],
    [31.8778, 0.9695, 0.4793],
    [31.1553, 0.9640, 0.4173],
    [30.3977, 0.9571, 0.3766],
    [29.6014, 0.9484, 0.3502],
    [28.8030, 0.9379, 0.3332]
])

CConv_bypass_high_ld_5e_3 = RD_Curve()
CConv_bypass_high_ld_5e_3.add_points([
    # [30.8369, 0.9617, 0.5636],
    # [30.2025, 0.9557, 0.4113],
    # [29.4788, 0.9476, 0.3149],
    # [28.7021, 0.9372, 0.2556],
    # [27.9054, 0.9243, 0.2199],
    # [27.1077, 0.9083, 0.1988]
    [30.6536, 0.9597, 0.3815],
    [30.0520, 0.9536, 0.3153],
    [29.3979, 0.9459, 0.2720],
    [28.6921, 0.9363, 0.2435],
    [27.9775, 0.9248, 0.2254],
    [27.2378, 0.9104, 0.2140]
])

CConv_bypass_high_ld_1e_3 = RD_Curve()
CConv_bypass_high_ld_1e_3.add_points([
    # [26.7310, 0.9052, 0.1896],
    # [26.3336, 0.8939, 0.1416],
    # [25.8914, 0.8798, 0.1122],
    # [25.3967, 0.8622, 0.0946],
    # [24.8780, 0.8419, 0.0842],
    # [24.3036, 0.8171, 0.0781]
    [26.8894, 0.9027, 0.1344],
    [26.4236, 0.8914, 0.1108],
    [25.9056, 0.8776, 0.0957],
    [25.3409, 0.8602, 0.0861],
    [24.6823, 0.8396, 0.0801],
    [23.9529, 0.8136, 0.0764]
])

CConv_bypass_high_3 = RD_Curve()
CConv_bypass_high_3.add_points([
    [38.4623, 0.9924, 1.3794],  # 0524_2343
    [36.4283, 0.9884, 1.0190],
    [31.7308, 0.9679, 0.4408],
    [29.8731, 0.9512, 0.2923],
    [26.2567, 0.8863, 0.1057],
])

Google_blur_5 = RD_Curve()
Google_blur_5.add_points([
    [38.4715, 0.9941, 2.2207],
    [35.8581, 0.9859, 1.3933],
    [33.3514, 0.9754, 0.7853],
    [29.3601, 0.9401, 0.3525]
])

Google_blur_3 = RD_Curve()
Google_blur_3.add_points([
    [38.4801, 0.9941, 2.2157],
    [35.8740, 0.9860, 1.3933],
    [33.3384, 0.9753, 0.7863],
    [29.1343, 0.9379, 0.3419]
])

Google_blur_1 = RD_Curve()
Google_blur_1.add_points([
    [38.4780, 0.9941, 2.2163],
    [35.8616, 0.9859, 1.3920],
    [33.3340, 0.9752, 0.7853],
    [29.1119, 0.9376, 0.3411]
])

Google_blur_0 = RD_Curve()
Google_blur_0.add_points([
    [38.4772, 0.9941, 2.2155],
    [35.8613, 0.9859, 1.3922],
    [33.3336, 0.9752, 0.7853],
    [29.0475, 0.9368, 0.3391]
])

Blur_0_pretrain = RD_Curve()
Blur_0_pretrain.add_points([
    [39.2055, 0.9960, 2.2031],
    [37.5121, 0.9922, 1.3141],
    [36.1966, 0.9887, 1.0018],
    [31.7508, 0.9670, 0.4716]
])

Blur_4_pretrain = RD_Curve()
Blur_4_pretrain.add_points([
    [39.0705, 0.9958, 2.1964],
    [37.4910, 0.9922, 1.3134],
    [36.1582, 0.9886, 1.0000],
    [31.7381, 0.9672, 0.4663]
])

Blur_0_from_scratch = RD_Curve()
Blur_0_from_scratch.add_points([
    [38.8269, 0.9955, 2.3755],
    [37.3850, 0.9924, 1.4562],
    [36.2672, 0.9895, 1.0975],
    [32.0846, 0.9686, 0.4762],
])

Blur_4_from_scratch = RD_Curve()
Blur_4_from_scratch.add_points([
    [38.2066, 0.9948, 2.3294],
    [36.9144, 0.9919, 1.4069],
    [36.0542, 0.9890, 1.0723],
    [31.8798, 0.9679, 0.4604],
])



fig_plot = RD_Plot()

fig_plot.add_curve(BPG.bpp, BPG.PSNR, color='black', label='BPG')
# fig_plot.add_curve(FM2_wPCA_wRE.bpp, FM2_wPCA_wRE.PSNR, color='blue', label='FM2 w/ PCA')
# fig_plot.add_curve(FM4_wPCA_wRE.bpp, FM4_wPCA_wRE.PSNR, color='green', label='FM4 w/ PCA')
# fig_plot.add_curve(FM8_wPCA_wRE.bpp, FM8_wPCA_wRE.PSNR, color='red', label='FM8 w/ PCA')

fig_plot.add_curve(Google_inference.bpp, Google_inference.PSNR, label='Google paper RD')

# fig_plot.add_curve(Google_reRun.bpp, Google_reRun.PSNR, label='HP coder TensorFlow')
fig_plot.add_curve(Google_pytorch.bpp, Google_pytorch.PSNR, label='HP coder Pytorch')
# fig_plot.add_curve(Google_pytorch_3.bpp, Google_pytorch_3.PSNR, label='HP coder Pytorch 3 days')

fig_plot.add_curve(Blur_0_pretrain.bpp, Blur_0_pretrain.PSNR, label='Blur x4 pre-train', marker='x')
fig_plot.add_curve(Blur_4_pretrain.bpp, Blur_4_pretrain.PSNR, label='Blur x0 pre-train', marker='x')
fig_plot.add_curve(Blur_0_from_scratch.bpp, Blur_0_from_scratch.PSNR, label='Blur x4 from scratch', marker='x')
fig_plot.add_curve(Blur_4_from_scratch.bpp, Blur_4_from_scratch.PSNR, label='Blur x0 from scratch', marker='x')

fig_plot.plot()
